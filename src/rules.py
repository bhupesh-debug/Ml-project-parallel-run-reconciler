"""Rules-first baseline. Ordered checks a reviewer could apply by hand.

Precedence (later checks override earlier ones):

1. default: ``rule`` — an unexplained difference is most likely a logic change.
2. amount changed or missing -> ``data``.
3. new commission is 0 and the deal crosses a period boundary -> ``timing``.
4. deal missing from the new system entirely -> ``data``. This deliberately
   overrides check 3: per the project brief, a missing record is a *data*
   cause (a feed/migration gap to flag), never a timing difference, even if
   the deal also happens to cross a period boundary.
5. |diff| within the rounding tolerance -> ``rounding``.
6. exact match -> ``none``.

The rounding tolerance is configurable (``cent_tolerance``).
"""
import numpy as np
import pandas as pd


def classify_rules(df: pd.DataFrame, cent_tolerance: float = 0.05) -> np.ndarray:
    diff = df["new_commission"] - df["old_commission"]
    out = np.full(len(df), "rule", dtype=object)  # default for unexplained differences
    amount_changed = df["amount_new"].isna() | (df["amount_new"] != df["amount_old"])
    missing_new = df["amount_new"].isna()
    new_zero = df["new_commission"] == 0

    out[amount_changed.to_numpy()] = "data"
    out[(new_zero & (df["crosses_period"] == 1)).to_numpy()] = "timing"
    out[missing_new.to_numpy()] = "data"  # missing record: data cause, not timing
    out[(diff.abs() <= cent_tolerance).to_numpy()] = "rounding"
    out[(diff == 0).to_numpy()] = "none"
    return out
