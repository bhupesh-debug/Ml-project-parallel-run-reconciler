"""The human-in-the-loop layer: cascade, reasons, ranked review lists.

Design (the brief's differentiation layer): **rules first, ML second.**

1. Exact matches and cent-level differences are cleared by rules a finance
   reviewer can read and audit.
2. A deal missing from the new system is flagged by rule as a *data* cause —
   a feed/migration gap, not a calculation error.
3. Everything else goes to the gradient-boosting model. Items at or above
   the confidence threshold are auto-classified; the rest are marked
   ``needs_review`` and ranked to the top of the investigation list with the
   model's top two candidate causes shown — so overlapping dual causes stay
   visible instead of being collapsed to one label.

A person still reviews every item. This module ranks and explains; it never
posts a change.
"""
import numpy as np
import pandas as pd


def reason_for_deal(row: pd.Series, label: str, cent_tolerance: float = 0.05) -> str:
    """Plain-language reason a sales rep or finance reviewer can understand."""
    diff = float(row["new_commission"] - row["old_commission"])
    if label == "none":
        return "Old and new commissions match exactly — nothing to resolve."
    if pd.isna(row["amount_new"]):
        return ("Deal is missing from the new system entirely. Treat as a "
                "data/feed gap to flag with the migration team, not a "
                "calculation error.")
    if label == "rounding":
        return (f"Difference of {diff:+.2f} is inside the {cent_tolerance:.2f} "
                "rounding tolerance — cent-level rounding between the two "
                "calculations. Safe to clear in bulk.")
    if label == "timing":
        return ("New commission is 0.00 and the deal crosses a period "
                "boundary — the commission most likely landed in the next "
                "period in the new system.")
    if label == "data":
        return (f"Deal amount changed between systems "
                f"({float(row['amount_old']):,.2f} -> {float(row['amount_new']):,.2f}), "
                "so the source data differs — check the feed or the record "
                "itself before questioning the calculation.")
    if label == "rule":
        ratio = (float(row["new_commission"]) / float(row["old_commission"])
                 if float(row["old_commission"]) else 0.0)
        return (f"New commission is {ratio:.0%} of the old one and the gap "
                "scales with deal size — consistent with a rate or plan-logic "
                "change in the new system. Confirm against the new plan rules.")
    return f"Predicted cause: {label}."


def cascade_predict(df: pd.DataFrame, proba: np.ndarray, classes,
                    threshold: float = 0.9, cent_tolerance: float = 0.05):
    """Apply rules first, ML second. Returns (labels, sources, confidence).

    ``sources`` is one of ``rules`` / ``ml`` / ``needs_review`` per row.
    """
    cls = np.array(classes)
    top = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    labels = cls[top].astype(object)
    sources = np.where(conf >= threshold, "ml", "needs_review").astype(object)

    diff = (df["new_commission"] - df["old_commission"]).to_numpy()
    missing = df["amount_new"].isna().to_numpy()
    for i in range(len(df)):
        if diff[i] == 0:
            labels[i], sources[i], conf[i] = "none", "rules", 1.0
        elif missing[i]:
            labels[i], sources[i], conf[i] = "data", "rules", 1.0
        elif abs(diff[i]) <= cent_tolerance:
            labels[i], sources[i], conf[i] = "rounding", "rules", 1.0
    return labels, sources, conf


def build_review_list(df: pd.DataFrame, rules_pred, proba, classes,
                      threshold: float = 0.9,
                      cent_tolerance: float = 0.05) -> pd.DataFrame:
    """Ranked investigation list for Problem A (one row per differing deal).

    Ranking: ``needs_review`` items first (a person must decide), then
    ML-auto-classified, then rule-cleared; within each group, larger dollar
    gaps first, with low confidence pushing an item up.
    ``priority_score = |diff| * (1 + (1 - confidence))``.
    """
    cls = np.array(classes)
    order = np.argsort(-proba, axis=1)
    labels, sources, conf = cascade_predict(df, proba, classes, threshold, cent_tolerance)

    out = pd.DataFrame({
        "deal_id": df["deal_id"].to_numpy(),
        "close_month": df["close_month"].to_numpy(),
        "plan": df["plan"].to_numpy(),
        "amount_old": df["amount_old"].to_numpy(),
        "amount_new": df["amount_new"].to_numpy(),
        "old_commission": df["old_commission"].to_numpy(),
        "new_commission": df["new_commission"].to_numpy(),
        "diff_amount": (df["new_commission"] - df["old_commission"]).to_numpy(),
        "missing_in_new": df["amount_new"].isna().to_numpy(),
        "rules_label": np.asarray(rules_pred),
        "predicted_label": labels,
        "confidence": np.round(conf, 4),
        "second_label": cls[order[:, 1]],
        "second_confidence": np.round(proba[np.arange(len(df)), order[:, 1]], 4),
        "source": sources,
        "injected_cause": df["cause"].to_numpy(),
        "injected_secondary": df["cause_secondary"].to_numpy(),
    })
    out["abs_diff"] = out["diff_amount"].abs()
    out["priority_score"] = (out["abs_diff"] * (1 + (1 - out["confidence"]))).round(2)
    df_reset = df.reset_index(drop=True)
    out["reason"] = [reason_for_deal(df_reset.iloc[i], out["predicted_label"].iloc[i],
                                     cent_tolerance)
                     for i in range(len(df_reset))]

    group_rank = {"needs_review": 0, "ml": 1, "rules": 2}
    out["_group"] = out["source"].map(group_rank)
    out = out.sort_values(["_group", "priority_score"],
                          ascending=[True, False]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.drop(columns=["_group", "abs_diff"])


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(str(a).lower().split()), set(str(b).lower().split())
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def mapping_reason(row: pd.Series) -> str:
    """Plain-language reason for a flagged account mapping."""
    bits = []
    if row["category"] != row["mapped_category"]:
        bits.append(f"mapped code belongs to {row['mapped_category']}, "
                    f"not {row['category']}")
    if _token_jaccard(row["account_name"], row["mapped_account_name"]) < 0.34:
        bits.append(f"mapped code's account is '{row['mapped_account_name']}', "
                    "which does not match this account's name")
    if row["collision_count"] > 1:
        bits.append(f"{int(row['collision_count'])} source accounts post "
                    "into this one new code")
    old, new = float(row["old_balance"]), float(row["new_balance"])
    if old != 0 and abs(new / old - 1) > 0.25:
        bits.append(f"new balance is {new / old:.2f}x the old balance")
    if (old > 0) != (new > 0) and old != 0 and new != 0:
        bits.append("balance sign flipped between systems")
    if not bits:
        bits.append("balance pattern is an outlier vs the account's own history")
    text = "; ".join(bits)
    return text[:1].upper() + text[1:] + "."


def build_mapping_review_list(df: pd.DataFrame, proba_mis: np.ndarray,
                              features: pd.DataFrame) -> pd.DataFrame:
    """Ranked short review list for Problem B, most suspicious first."""
    feats = features.reset_index(drop=True)
    out = pd.DataFrame({
        "account_id": df["account_id"].to_numpy(),
        "account_name": df["account_name"].to_numpy(),
        "category": df["category"].to_numpy(),
        "entity": df["entity"].to_numpy(),
        "old_code": df["old_code"].to_numpy(),
        "mapped_new_code": df["mapped_new_code"].to_numpy(),
        "mapped_account_name": df["mapped_account_name"].to_numpy(),
        "mapped_category": df["mapped_category"].to_numpy(),
        "old_balance": df["old_balance"].to_numpy(),
        "new_balance": df["new_balance"].to_numpy(),
        "collision_count": feats["collision_count"].to_numpy(),
        "mismap_probability": np.round(proba_mis, 4),
        "is_mismapped": df["is_mismapped"].to_numpy(),
    })
    out["reason"] = [mapping_reason(out.iloc[i]) for i in range(len(out))]
    out = out.sort_values("mismap_probability", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out
