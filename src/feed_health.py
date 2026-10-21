"""Feed-health monitor (Problem C, lightweight).

Scores each month of a data feed from three operational signals — row-count
completeness, null rate, and late arrivals — into a 0-100 health score.
Months below the break threshold are flagged so a pipeline break is caught
from the trend, not from a failed reconciliation downstream.

The synthetic series injects three breaks: a row-count drop, a null spike,
and a late-arrival spike. All data is synthetic.
"""
import numpy as np
import pandas as pd

BREAK_THRESHOLD = 75.0
WARN_THRESHOLD = 90.0


def health_score(completeness: float, null_rate: float, late_share: float) -> float:
    """0-100 score. Penalties: up to 60 for missing rows, up to 40 for nulls
    (saturating at a 5% null rate), up to 30 for late arrivals (saturating
    at a 20% late share). A normal month (<=1.5% row shortfall, <=0.8%
    nulls, <=1% late) scores ~91+; any single injected break costs 27+
    points and lands below the break threshold."""
    score = 100.0
    score -= 60.0 * max(0.0, 1.0 - completeness)
    score -= 40.0 * min(null_rate / 0.05, 1.0)
    score -= 30.0 * min(late_share / 0.20, 1.0)
    return float(np.clip(score, 0.0, 100.0))


def status_for(score: float) -> str:
    if score >= WARN_THRESHOLD:
        return "healthy"
    if score >= BREAK_THRESHOLD:
        return "warning"
    return "break"


def generate_feed(n_months: int = 24, seed: int = 42) -> pd.DataFrame:
    """Monthly feed statistics with three injected breaks (months 7, 13, 19)."""
    rng = np.random.default_rng(seed)
    months = np.arange(1, n_months + 1)
    expected = np.round(50_000 * (1.012 ** months) * rng.uniform(0.98, 1.02, n_months)).astype(int)
    received = np.round(expected * rng.uniform(0.985, 1.0, n_months)).astype(int)
    null_rate = rng.uniform(0.001, 0.008, n_months)
    late = np.round(received * rng.uniform(0.002, 0.01, n_months)).astype(int)

    if n_months >= 7:
        received[6] = int(expected[6] * 0.55)          # break 1: row drop
    if n_months >= 13:
        null_rate[12] = 0.08                            # break 2: null spike
    if n_months >= 19:
        late[18] = int(received[18] * 0.25)             # break 3: late arrivals

    df = pd.DataFrame({
        "month_index": months,
        "month": [f"2025-{((m - 1) % 12) + 1:02d}" if m <= 12 else f"2026-{((m - 1) % 12) + 1:02d}"
                  for m in months],
        "rows_expected": expected,
        "rows_received": received,
        "null_rate": np.round(null_rate, 4),
        "late_arrivals": late,
    })
    return score_feed(df)


def score_feed(df: pd.DataFrame) -> pd.DataFrame:
    """Add completeness, late share, health score and status to a feed table."""
    out = df.copy()
    out["completeness"] = out["rows_received"] / out["rows_expected"]
    out["late_share"] = out["late_arrivals"] / out["rows_received"]
    out["health_score"] = [health_score(c, n, l) for c, n, l in
                            zip(out["completeness"], out["null_rate"], out["late_share"])]
    out["status"] = [status_for(s) for s in out["health_score"]]
    return out


if __name__ == "__main__":
    feed = generate_feed()
    feed.to_csv("data/synthetic/feed_health.csv", index=False)
    print(feed[["month", "rows_expected", "rows_received", "null_rate",
                "late_arrivals", "health_score", "status"]].to_string(index=False))
