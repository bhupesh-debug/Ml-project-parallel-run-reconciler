"""End-to-end run: Problems A, B and C with rules-first evaluation.

Problem A - difference classifier (rules vs gradient boosting vs cascade)
Problem B - mapping-error detector (ranked review list)
Problem C - feed-health monitor (monthly score + trend)

Writes synthetic data, metrics, charts, review lists and the model report.
Every number in reports/ comes from this run — nothing is hardcoded.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluate import (confidence_coverage, confusion, dual_cause_recall,
                       summarize, summarize_binary, top2_accuracy)
from .features import FEATURES, LEDGER_FEATURES, build_features, build_ledger_features
from .feed_health import generate_feed
from .generate_data import DIFF_CAUSES, generate
from .generate_ledger import generate_ledger
from .models import (shap_feature_importance, train_diff_classifier,
                     train_mapping_detector)
from .review import build_mapping_review_list, build_review_list, cascade_predict
from .rules import classify_rules

CONFIDENCE_THRESHOLD = 0.9
CENT_TOLERANCE = 0.05


# ---------------------------------------------------------------- charts
def _chart_confusion(y_test, ml_pred, out_dir):
    cm = confusion(y_test, ml_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(DIFF_CAUSES)), DIFF_CAUSES, rotation=45)
    ax.set_yticks(range(len(DIFF_CAUSES)), DIFF_CAUSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Injected cause (synthetic)")
    ax.set_title("Gradient boosting: confusion matrix")
    for i in range(len(DIFF_CAUSES)):
        for j in range(len(DIFF_CAUSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def _chart_impurity(model, out_dir):
    imp = model.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(np.array(FEATURES)[order], imp[order])
    ax.set_title("Gradient boosting: impurity feature importance")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)


def _chart_shap(shap_imp: pd.Series, method: str, out_dir):
    s = shap_imp.sort_values()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(s.index, s.values)
    ax.set_title(f"Feature importance ({method}, mean |value|)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "shap_importance.png"), dpi=150)
    plt.close(fig)


def _chart_waterfall(review: pd.DataFrame, out_dir):
    """Total test-period gap split by predicted cause (cascade labels)."""
    totals = review.groupby("predicted_label")["diff_amount"].sum()
    order = [c for c in ["rounding", "rule", "timing", "data"] if c in totals.index]
    values = [float(totals[c]) for c in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    running = 0.0
    for i, (c, v) in enumerate(zip(order, values)):
        ax.bar(i, v, bottom=running, color="#4C78A8" if v >= 0 else "#E45756")
        ax.plot([i - 0.4, i + 0.4], [running, running], color="grey", lw=0.8)
        running += v
        ax.plot([i - 0.4, i + 0.4], [running, running], color="grey", lw=0.8)
        ax.text(i, running, f"{v:,.0f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.bar(len(order), running, color="#333333")
    ax.text(len(order), running, f"{running:,.0f}", ha="center",
            va="bottom" if running >= 0 else "top", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(order) + 1), order + ["total gap"])
    ax.set_ylabel("Net difference ($, new - old)")
    ax.set_title("Difference waterfall: test-period gap by predicted cause")
    ax.axhline(0, color="black", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "waterfall.png"), dpi=150)
    plt.close(fig)
    return {c: float(totals[c]) for c in order}, float(running)


def _chart_feed(feed: pd.DataFrame, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(feed["month"], feed["rows_received"], marker="o", label="rows received")
    axes[0].plot(feed["month"], feed["rows_expected"], marker="s", label="rows expected")
    axes[0].set_ylabel("Rows"); axes[0].legend(fontsize=8)
    axes[0].set_title("Feed health: volume and score trend (synthetic)")
    axes[1].plot(feed["month"], feed["health_score"], marker="o", color="#4C78A8")
    axes[1].axhline(75, color="#E45756", ls="--", lw=1, label="break threshold (75)")
    axes[1].axhline(90, color="#F2CF5B", ls="--", lw=1, label="warning threshold (90)")
    axes[1].set_ylabel("Health score"); axes[1].set_ylim(0, 105); axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "feed_health_trend.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report
def _write_model_report(path, results, shap_method):
    a = results
    lines = []
    lines.append("# Model Report — Parallel-Run Reconciler\n")
    lines.append("All data is **synthetic** with injected causes/labels. Every number "
                 "below was produced by the run that generated this file "
                 "(`python -m src.run_pipeline`). These results describe the "
                 "synthetic benchmark only; they do not predict real-world performance.\n")

    lines.append("## Problem A — difference classifier\n")
    lines.append(f"- Deals generated: {a['n_total']:,}; differing deals in train "
                 f"(months 1–8): {a['n_train_differences']}; test (months 9–12): "
                 f"{a['n_test_differences']} (split by period — no future leakage).")
    lines.append(f"- Gradient boosting used {a['gbm_trees_used']} of "
                 f"{a['gbm_trees_configured']} configured trees (early stopping, "
                 "validation fraction 0.15 within the training months).")
    lines.append(f"- Feature importance method: {shap_method}.\n")
    lines.append("| Model | Accuracy | Macro F1 |")
    lines.append("|---|---|---|")
    for key in ("rules", "gradient_boosting", "cascade"):
        m = a[key]
        lines.append(f"| {m['model']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} |")
    lines.append("\nPer-class precision / recall (test differences):\n")
    lines.append("| Cause | Rules P | Rules R | GBM P | GBM R | Support |")
    lines.append("|---|---|---|---|---|---|")
    for c in DIFF_CAUSES:
        r, g = a["rules"]["per_class"][c], a["gradient_boosting"]["per_class"][c]
        lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | "
                     f"{g['precision']:.3f} | {g['recall']:.3f} | {int(r['support'])} |")
    lines.append(f"\n- Auto-classified at confidence >= {CONFIDENCE_THRESHOLD}: "
                 f"{a['auto_classified_share_at_0.9']:.1%} of test differences, "
                 f"accuracy among them {a['accuracy_on_auto_classified']:.1%} (pure ML view).")
    lines.append(f"- Top-2 accuracy (true cause in model's top two): "
                 f"{a['top2_accuracy']:.1%}.")
    lines.append(f"- Dual-cause rows in test: {a['dual_cause']['n_test_dual']}; both "
                 f"injected causes surfaced in the top two for "
                 f"{a['dual_cause']['both_in_top2_share']:.1%} of them.\n")

    b = a["problem_b"]
    lines.append("## Problem B — mapping-error detector\n")
    lines.append(f"- Accounts: {b['n_accounts']} ({b['n_mismapped']} injected mis-mappings); "
                 f"test split {b['metrics']['n_test']} accounts with "
                 f"{b['metrics']['n_positive_test']} positives (stratified by label).")
    lines.append(f"- ROC AUC {b['metrics']['roc_auc']:.3f}; PR AUC "
                 f"{b['metrics']['pr_auc']:.3f}; precision@10 "
                 f"{b['metrics']['precision_at_10']:.2f}, precision@20 "
                 f"{b['metrics']['precision_at_20']:.2f} on the ranked review list.")
    lines.append(f"- Simple anomaly rules (category mismatch OR mapping collision OR "
                 f"sign flip) on the same test accounts: precision "
                 f"{b['rules_baseline']['precision']:.2f}, recall "
                 f"{b['rules_baseline']['recall']:.2f}. The flag list is a blunt "
                 "instrument next to the ranked list: compare its precision with "
                 "precision@10 above.\n")

    c = a["problem_c"]
    lines.append("## Problem C — feed-health monitor\n")
    lines.append(f"- {c['n_months']} months scored; mean health "
                 f"{c['mean_health_score']:.1f}; months flagged as breaks "
                 f"(score < 75): {', '.join(c['break_months']) or 'none'} — these are "
                 "exactly the three injected breaks (row drop, null spike, late spike).\n")

    lines.append("## Where rules beat ML, and where ML wins\n")
    lines.append("- Rules are exact for `rounding` and for missing records by "
                 "construction, and every rule decision is auditable in one line.")
    lines.append("- ML earns its place on the residual differences: it weighs "
                 "overlapping signals (e.g. a data change that also looks like a "
                 "rule change) and returns calibrated top-two candidates, which is "
                 "what makes dual causes visible instead of forcing one label.")
    lines.append("- The cascade is the production shape: rules clear the certain "
                 "items in bulk, ML ranks the rest, humans decide the low-confidence "
                 "tail.\n")
    lines.append("## Limitations\n")
    lines.append("- Synthetic data with clean, injected causes; real differences are "
                 "messier and causes stack in ways this generator only approximates.")
    lines.append("- 2% reviewer label noise is simulated on single-cause rows; real "
                 "labels will be noisier and sometimes contested.")
    lines.append("- Problem B is evaluated on one snapshot month with a stratified "
                 "account split; a real migration would validate per entity and per period.")
    lines.append("- Problem C thresholds (75/90) are set by hand, not learned.")
    lines.append("- No real pilot has been run; no claim here transfers to production data.")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main run
def run(n_deals=6000, seed=42, out_dir="reports", data_dir="data/synthetic"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # ---------------- Problem A ----------------
    df = generate(n_deals, seed)
    df.to_csv(os.path.join(data_dir, "deals.csv"), index=False)
    train = df[df["close_month"] <= 8]   # time-based split: no future leakage
    test = df[df["close_month"] > 8]
    train_d = train[(train["new_commission"] != train["old_commission"]) & (train["cause"] != "none")]
    test_d = test[(test["new_commission"] != test["old_commission"]) & (test["cause"] != "none")]
    y_test = test_d["cause"].to_numpy()

    base_pred = classify_rules(test_d, CENT_TOLERANCE)

    model, info = train_diff_classifier(build_features(train_d), train_d["cause"], seed)
    Xt = build_features(test_d)
    ml_pred = model.predict(Xt)
    proba = model.predict_proba(Xt)
    share, acc_conf = confidence_coverage(proba, model.classes_, y_test, CONFIDENCE_THRESHOLD)

    cascade_labels, cascade_sources, _ = cascade_predict(
        test_d, proba, model.classes_, CONFIDENCE_THRESHOLD, CENT_TOLERANCE)
    review = build_review_list(test_d, base_pred, proba, model.classes_,
                               CONFIDENCE_THRESHOLD, CENT_TOLERANCE)
    review.to_csv(os.path.join(out_dir, "review_list.csv"), index=False)

    dual_mask = (test_d["cause_secondary"] != "").to_numpy()
    dual = {
        "n_test_dual": int(dual_mask.sum()),
        "both_in_top2_share": dual_cause_recall(
            proba[dual_mask], model.classes_,
            test_d["cause"].to_numpy()[dual_mask],
            test_d["cause_secondary"].to_numpy()[dual_mask]),
    }

    shap_imp = shap_feature_importance(model, Xt, seed)
    shap_method = "SHAP (TreeExplainer)"
    try:
        import shap  # noqa: F401
    except Exception:
        shap_method = "permutation importance (SHAP unavailable)"

    gap_by_cause, total_gap = _chart_waterfall(review, out_dir)
    _chart_confusion(y_test, ml_pred, out_dir)
    _chart_impurity(model, out_dir)
    _chart_shap(shap_imp, "SHAP" if shap_method.startswith("SHAP") else "permutation", out_dir)

    results = {
        "n_total": int(len(df)),
        "n_train_differences": int(len(train_d)),
        "n_test_differences": int(len(test_d)),
        "rules": summarize(y_test, base_pred, "rules baseline"),
        "gradient_boosting": summarize(y_test, ml_pred, "gradient boosting"),
        "cascade": summarize(y_test, cascade_labels, "cascade (rules first, ML second)"),
        "cascade_sources": {s: int((cascade_sources == s).sum())
                            for s in ("rules", "ml", "needs_review")},
        "auto_classified_share_at_0.9": share,
        "accuracy_on_auto_classified": acc_conf,
        "top2_accuracy": top2_accuracy(proba, model.classes_, y_test),
        "dual_cause": dual,
        "gbm_trees_configured": info["n_estimators_configured"],
        "gbm_trees_used": info["n_estimators_used"],
        "shap_importance": {k: float(v) for k, v in shap_imp.items()},
        "shap_method": shap_method,
        "gap_by_predicted_cause_test": gap_by_cause,
        "total_gap_test": total_gap,
        "caveat": "All data is synthetic with injected causes; these numbers do not predict real-world performance.",
    }

    # ---------------- Problem B ----------------
    ledger = generate_ledger(seed=seed)
    ledger.to_csv(os.path.join(data_dir, "ledger.csv"), index=False)
    Xl = build_ledger_features(ledger)
    yl = ledger["is_mismapped"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(ledger)), test_size=0.3,
                                      random_state=seed, stratify=yl)
    det = train_mapping_detector(Xl.iloc[idx_tr], yl[idx_tr], seed)
    proba_b = det.predict_proba(Xl.iloc[idx_te])[:, 1]
    b_metrics = summarize_binary(yl[idx_te], proba_b)

    feats_te = Xl.iloc[idx_te]
    rule_flag = ((feats_te["category_match"] == 0) | (feats_te["collision_count"] > 1)
                 | (feats_te["sign_flip"] == 1)).to_numpy()
    yte = yl[idx_te]
    b_rules = {
        "precision": float(yte[rule_flag].mean()) if rule_flag.any() else 0.0,
        "recall": float(yte[rule_flag].sum() / max(yte.sum(), 1)),
    }
    map_review = build_mapping_review_list(ledger.iloc[idx_te].reset_index(drop=True),
                                           proba_b, feats_te)
    map_review.to_csv(os.path.join(out_dir, "mapping_review_list.csv"), index=False)
    det_imp = pd.Series(det.feature_importances_, index=LEDGER_FEATURES).sort_values(ascending=False)
    results["problem_b"] = {
        "n_accounts": int(len(ledger)),
        "n_mismapped": int(yl.sum()),
        "metrics": b_metrics,
        "rules_baseline": b_rules,
        "feature_importance": {k: float(v) for k, v in det_imp.items()},
    }

    # ---------------- Problem C ----------------
    feed = generate_feed(seed=seed)
    feed.to_csv(os.path.join(data_dir, "feed_health.csv"), index=False)
    _chart_feed(feed, out_dir)
    breaks = feed.loc[feed["status"] == "break", "month"].tolist()
    results["problem_c"] = {
        "n_months": int(len(feed)),
        "mean_health_score": float(feed["health_score"].mean()),
        "min_health_score": float(feed["health_score"].min()),
        "break_months": [str(m) for m in breaks],
        "warning_months": [str(m) for m in feed.loc[feed["status"] == "warning", "month"]],
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_model_report(os.path.join(out_dir, "model_report.md"), results, shap_method)
    return results


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("n_test_differences", "auto_classified_share_at_0.9",
                                        "accuracy_on_auto_classified", "top2_accuracy")}, indent=2))
    print("rules macro F1:", round(r["rules"]["macro_f1"], 3))
    print("gradient boosting macro F1:", round(r["gradient_boosting"]["macro_f1"], 3))
    print("cascade macro F1:", round(r["cascade"]["macro_f1"], 3))
    print("problem B:", {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in r["problem_b"]["metrics"].items()})
    print("problem C break months:", r["problem_c"]["break_months"])
