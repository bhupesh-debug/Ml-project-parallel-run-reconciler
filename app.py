"""Streamlit review app for the Parallel-Run Reconciler.

Loads the pipeline outputs (reports/) and presents the ranked investigation
list with plain-language reasons — the human-in-the-loop review sheet from
the project architecture. Run the pipeline first:

    python -m src.run_pipeline
    streamlit run app.py

All data is synthetic.
"""
import json
import os

import pandas as pd
import streamlit as st

REPORTS = "reports"
DATA = os.path.join("data", "synthetic")

st.set_page_config(page_title="Parallel-Run Reconciler", layout="wide")
st.title("Parallel-Run Reconciler — Review Sheet")
st.caption("Old vs new system differences, classified by likely cause and "
           "ranked for investigation. All data is synthetic; a person reviews "
           "every item — this app ranks and explains, it never posts a change.")

metrics_path = os.path.join(REPORTS, "metrics.json")
review_path = os.path.join(REPORTS, "review_list.csv")
if not (os.path.exists(metrics_path) and os.path.exists(review_path)):
    st.warning("No pipeline outputs found. Run `python -m src.run_pipeline` "
               "first, then reload this app.")
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)
review = pd.read_csv(review_path)

# ---------------------------------------------------------------- Problem A
st.header("Problem A — commission differences (test period)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rules macro F1", f"{metrics['rules']['macro_f1']:.3f}")
c2.metric("Gradient boosting macro F1", f"{metrics['gradient_boosting']['macro_f1']:.3f}")
c3.metric("Cascade macro F1", f"{metrics['cascade']['macro_f1']:.3f}")
c4.metric("Auto-classified @ 0.90 confidence",
          f"{metrics['auto_classified_share_at_0.9']:.0%}")

src = metrics.get("cascade_sources", {})
st.write(f"Cascade outcome on {metrics['n_test_differences']} test differences: "
         f"**{src.get('rules', 0)} cleared by rules**, "
         f"**{src.get('ml', 0)} auto-classified by ML**, "
         f"**{src.get('needs_review', 0)} need a human decision** (ranked first below).")

causes = ["all"] + sorted(review["predicted_label"].unique().tolist())
sources = ["all"] + sorted(review["source"].unique().tolist())
f1, f2, f3 = st.columns(3)
cause_pick = f1.selectbox("Predicted cause", causes)
source_pick = f2.selectbox("Decision source", sources)
top_n = f3.slider("Show top N by investigation rank", 10, len(review), min(50, len(review)))

view = review.copy()
if cause_pick != "all":
    view = view[view["predicted_label"] == cause_pick]
if source_pick != "all":
    view = view[view["source"] == source_pick]
view = view.head(top_n)
st.dataframe(view[["rank", "deal_id", "plan", "amount_old", "diff_amount",
                   "predicted_label", "confidence", "second_label",
                   "second_confidence", "source", "priority_score", "reason"]],
             width="stretch", hide_index=True)
st.caption("`injected_cause` columns are hidden here on purpose: in a real "
           "review the answer is not known. They are in the CSV for evaluation only.")

for img, title in (("waterfall.png", "Gap by predicted cause"),
                   ("confusion_matrix.png", "Confusion matrix (ML)"),
                   ("shap_importance.png", "Feature importance (SHAP)")):
    p = os.path.join(REPORTS, img)
    if os.path.exists(p):
        st.subheader(title)
        st.image(p)

# ---------------------------------------------------------------- Problem B
map_path = os.path.join(REPORTS, "mapping_review_list.csv")
if os.path.exists(map_path) and "problem_b" in metrics:
    st.header("Problem B — account mappings to check first")
    b = metrics["problem_b"]["metrics"]
    st.write(f"Detector ROC AUC **{b['roc_auc']:.3f}** · precision@10 "
             f"**{b['precision_at_10']:.2f}** · precision@20 **{b['precision_at_20']:.2f}**. "
             "Review from the top of this list down.")
    mrev = pd.read_csv(map_path)
    st.dataframe(mrev.head(25)[["rank", "account_name", "entity", "old_code",
                                 "mapped_new_code", "mapped_account_name",
                                 "old_balance", "new_balance",
                                 "mismap_probability", "reason"]],
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------- Problem C
feed_path = os.path.join(DATA, "feed_health.csv")
if os.path.exists(feed_path) and "problem_c" in metrics:
    st.header("Problem C — feed health")
    c = metrics["problem_c"]
    breaks = ", ".join(c["break_months"]) or "none"
    st.write(f"Mean health score **{c['mean_health_score']:.1f}** over "
             f"{c['n_months']} months. Months flagged as breaks: **{breaks}**.")
    img = os.path.join(REPORTS, "feed_health_trend.png")
    if os.path.exists(img):
        st.image(img)
    st.dataframe(pd.read_csv(feed_path), width="stretch", hide_index=True)
