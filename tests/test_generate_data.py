import numpy as np
import pandas as pd

from src.generate_data import CAUSES, DIFF_CAUSES, generate
from src.generate_ledger import ENTITIES, generate_ledger

EXPECTED_DEAL_COLUMNS = ["deal_id", "close_month", "close_date", "plan",
                         "amount_old", "amount_new", "old_commission",
                         "new_commission", "crosses_period", "cause",
                         "cause_secondary"]


def test_deals_schema_and_labels():
    df = generate(800, seed=11)
    assert list(df.columns) == EXPECTED_DEAL_COLUMNS
    assert set(df["cause"]).issubset(set(CAUSES))
    assert df["deal_id"].is_unique
    # close_date agrees with close_month
    months = pd.to_datetime(df["close_date"]).dt.month
    assert (months == df["close_month"]).all()


def test_none_means_exact_match_both_ways():
    df = generate(2000, seed=12)
    same = df["new_commission"] == df["old_commission"]
    assert (df.loc[same, "cause"] == "none").all()
    assert (df.loc[df["cause"] == "none", "new_commission"]
            == df.loc[df["cause"] == "none", "old_commission"]).all()


def test_missing_deals_have_zero_new_commission():
    df = generate(3000, seed=13)
    missing = df["amount_new"].isna()
    assert missing.sum() > 0
    assert (df.loc[missing, "new_commission"] == 0.0).all()


def test_dual_causes_exist_and_are_valid():
    df = generate(6000, seed=42)
    dual = df[df["cause_secondary"] != ""]
    assert len(dual) > 20  # ~5% of ~1080 differing deals
    assert set(dual["cause"]).issubset({"rule", "data"})
    assert set(dual["cause_secondary"]).issubset(set(DIFF_CAUSES))
    # rule + rounding: the gap is the rate gap plus at most 2 cents of noise
    rr = dual[dual["cause_secondary"] == "rounding"]
    rate_gap = rr["amount_old"] * 0.01
    actual = (rr["old_commission"] - rr["new_commission"]).abs()
    assert (np.abs(actual - rate_gap) <= 0.03).all()
    # data + rule: the amount really changed
    dr = dual[dual["cause_secondary"] == "rule"]
    assert (dr["amount_new"] != dr["amount_old"]).all()


def test_ledger_schema_labels_and_codes():
    led = generate_ledger(seed=7)
    assert led["new_code"].is_unique  # codes unique per entity chart
    share = led["is_mismapped"].mean()
    assert 0.08 <= share <= 0.18
    ok = led[led["is_mismapped"] == 0]
    bad = led[led["is_mismapped"] == 1]
    assert (ok["mapped_new_code"] == ok["new_code"]).all()
    assert (bad["mapped_new_code"] != bad["new_code"]).all()
    # mis-mappings stay inside the account's own entity chart
    assert (bad["entity"] == bad["mapped_new_code"].str.rsplit("-", n=1).str[-1]).all()
    assert set(led["entity"]) == set(ENTITIES)


def test_ledger_reproducible_and_renames_present():
    a, b = generate_ledger(seed=8), generate_ledger(seed=8)
    assert a.equals(b)
    renamed = a[a["new_account_name"] != a["account_name"]]
    assert len(renamed) > 0
    # a renamed but correctly mapped account points at its own new name
    ok_renamed = renamed[renamed["is_mismapped"] == 0]
    assert (ok_renamed["mapped_account_name"] == ok_renamed["new_account_name"]).all()
