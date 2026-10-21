"""Feature builders for Problem A (differences) and Problem B (mappings)."""
import numpy as np
import pandas as pd

FEATURES = [
    "diff", "abs_diff", "new_over_old", "amount_ratio", "amount_missing_new",
    "crosses_period", "new_is_zero", "is_cent_level", "amount_old", "plan_Standard",
    "plan_Tiered", "plan_Strategic",
]


def build_features(df: pd.DataFrame, cent_tolerance: float = 0.05) -> pd.DataFrame:
    """Difference features for the Problem A classifier (brief Step 4):
    diff size and sign, ratio to amount, period-boundary flag,
    missing-in-new flag, plan type."""
    X = pd.DataFrame(index=df.index)
    X["diff"] = df["new_commission"] - df["old_commission"]
    X["abs_diff"] = X["diff"].abs()
    X["new_over_old"] = np.where(df["old_commission"] > 0, df["new_commission"] / df["old_commission"], 1.0)
    amount_new = df["amount_new"].fillna(0.0)
    X["amount_ratio"] = np.where(df["amount_old"] > 0, amount_new / df["amount_old"], 1.0)
    X["amount_missing_new"] = df["amount_new"].isna().astype(int)
    X["crosses_period"] = df["crosses_period"]
    X["new_is_zero"] = (df["new_commission"] == 0).astype(int)
    X["is_cent_level"] = (X["abs_diff"] <= cent_tolerance).astype(int)
    X["amount_old"] = df["amount_old"]
    for p in ["Standard", "Tiered", "Strategic"]:
        X[f"plan_{p}"] = (df["plan"] == p).astype(int)
    return X[FEATURES]


LEDGER_FEATURES = [
    "balance_ratio", "abs_log_ratio", "sign_flip", "category_match",
    "name_similarity", "collision_count", "zero_new", "balance_magnitude",
    "balance_zscore",
]


def _token_jaccard(a: str, b: str) -> float:
    ta = set(str(a).lower().split())
    tb = set(str(b).lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_ledger_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anomaly features for the Problem B mapping-error detector.

    Everything here is visible to a reviewer at migration time: the old and
    new balances, the mapping table itself (collisions, category of the
    target code, the target account's name), and the account's own balance
    history. No label information is used.
    """
    X = pd.DataFrame(index=df.index)
    old = df["old_balance"].to_numpy(dtype=float)
    new = df["new_balance"].to_numpy(dtype=float)
    safe_old = np.where(old == 0, np.nan, old)
    ratio = new / safe_old
    X["balance_ratio"] = np.where(np.isnan(ratio), 0.0, ratio)
    X["abs_log_ratio"] = np.abs(np.log(np.where((ratio > 0) & np.isfinite(ratio), ratio, np.nan)))
    X["abs_log_ratio"] = X["abs_log_ratio"].fillna(2.0)  # sign flip / zero: maximally anomalous
    X["sign_flip"] = ((np.sign(old) != np.sign(new)) & (old != 0) & (new != 0)).astype(int)
    X["category_match"] = (df["category"] == df["mapped_category"]).astype(int)
    X["name_similarity"] = [_token_jaccard(a, b) for a, b in
                            zip(df["account_name"], df["mapped_account_name"])]
    counts = df["mapped_new_code"].value_counts()
    X["collision_count"] = df["mapped_new_code"].map(counts).astype(int)
    X["zero_new"] = (new == 0).astype(int)
    X["balance_magnitude"] = np.log1p(np.abs(old))
    std = df["old_balance_std"].to_numpy(dtype=float)
    mean = df["old_balance_mean"].to_numpy(dtype=float)
    X["balance_zscore"] = np.abs(new - mean) / np.where(std > 0, std, np.nan)
    X["balance_zscore"] = X["balance_zscore"].fillna(0.0).clip(upper=50.0)
    return X[LEDGER_FEATURES]
