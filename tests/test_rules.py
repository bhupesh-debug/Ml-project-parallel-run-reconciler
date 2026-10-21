import numpy as np
import pandas as pd

from src.rules import classify_rules


def _row(**kw):
    base = {"amount_old": 10_000.0, "amount_new": 10_000.0,
            "old_commission": 500.0, "new_commission": 500.0,
            "crosses_period": 0}
    base.update(kw)
    return pd.DataFrame([base])


def test_missing_deal_is_data_even_when_crossing_period():
    """Regression: the brief's edge case — a deal missing from the new
    system is a DATA cause, never timing, even if it crosses a period."""
    df = _row(amount_new=np.nan, new_commission=0.0, crosses_period=1)
    assert classify_rules(df)[0] == "data"


def test_timing_pattern():
    df = _row(new_commission=0.0, crosses_period=1)
    assert classify_rules(df)[0] == "timing"


def test_rounding_tolerance_is_configurable():
    df = _row(new_commission=500.03)  # 3-cent difference
    assert classify_rules(df, cent_tolerance=0.05)[0] == "rounding"
    assert classify_rules(df, cent_tolerance=0.02)[0] == "rule"


def test_amount_change_is_data():
    df = _row(amount_new=9_000.0, new_commission=450.0)
    assert classify_rules(df)[0] == "data"


def test_unexplained_difference_defaults_to_rule():
    df = _row(new_commission=450.0)  # same amount, no boundary, big gap
    assert classify_rules(df)[0] == "rule"
