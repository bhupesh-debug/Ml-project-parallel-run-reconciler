import numpy as np
import pandas as pd

from src.features import LEDGER_FEATURES, build_features, build_ledger_features
from src.generate_data import generate
from src.generate_ledger import generate_ledger


def test_diff_features_values():
    df = pd.DataFrame([{
        "amount_old": 1_000.0, "amount_new": np.nan, "old_commission": 50.0,
        "new_commission": 0.0, "crosses_period": 1, "plan": "Standard",
    }])
    X = build_features(df)
    row = X.iloc[0]
    assert row["diff"] == -50.0
    assert row["amount_missing_new"] == 1
    assert row["new_is_zero"] == 1
    assert row["amount_ratio"] == 0.0
    assert row["plan_Standard"] == 1


def test_ledger_features_shape_and_no_nans():
    led = generate_ledger(seed=3)
    X = build_ledger_features(led)
    assert list(X.columns) == LEDGER_FEATURES
    assert len(X) == len(led)
    assert not X.isna().any().any()


def test_ledger_features_for_correct_mapping():
    led = generate_ledger(seed=3)
    X = build_ledger_features(led)
    ok = led["is_mismapped"] == 0
    assert (X.loc[ok, "category_match"] == 1).all()
    # A correctly mapped account posts (almost) its own balance — unless
    # another account was mis-mapped INTO its code (a collision target),
    # which contaminates the posted balance through no fault of its own
    # mapping. Exclude collision targets before checking the ratio.
    clean = ok & (X["collision_count"] == 1)
    ratio = X.loc[clean, "balance_ratio"]
    assert ((ratio - 1).abs() < 0.01).all()
