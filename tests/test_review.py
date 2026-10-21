import numpy as np
import pandas as pd

from src.review import (build_mapping_review_list, build_review_list,
                        cascade_predict, reason_for_deal)
from src.rules import classify_rules

CLASSES = np.array(["data", "rounding", "rule", "timing"])


def _deals():
    return pd.DataFrame([
        # cent-level difference -> cleared by rules
        {"deal_id": "D-1", "close_month": 10, "plan": "Standard",
         "amount_old": 1000.0, "amount_new": 1000.0, "old_commission": 50.0,
         "new_commission": 50.01, "crosses_period": 0, "cause": "rounding",
         "cause_secondary": ""},
        # missing from new system -> data by rule, even crossing a period
        {"deal_id": "D-2", "close_month": 10, "plan": "Tiered",
         "amount_old": 2000.0, "amount_new": np.nan, "old_commission": 120.0,
         "new_commission": 0.0, "crosses_period": 1, "cause": "data",
         "cause_secondary": ""},
        # ambiguous big difference -> ML is unsure -> needs_review
        {"deal_id": "D-3", "close_month": 11, "plan": "Strategic",
         "amount_old": 9000.0, "amount_new": 9900.0, "old_commission": 360.0,
         "new_commission": 320.0, "crosses_period": 0, "cause": "data",
         "cause_secondary": "rule"},
    ])


def _proba():
    return np.array([
        [0.05, 0.90, 0.03, 0.02],   # ML agrees: rounding
        [0.97, 0.01, 0.01, 0.01],   # ML agrees: data
        [0.45, 0.05, 0.40, 0.10],   # ML unsure: data vs rule (the dual case)
    ])


def test_cascade_rules_first():
    df = _deals()
    labels, sources, conf = cascade_predict(df, _proba(), CLASSES, threshold=0.9)
    assert list(labels) == ["rounding", "data", "data"]
    assert list(sources) == ["rules", "rules", "needs_review"]
    assert conf[0] == 1.0 and conf[1] == 1.0


def test_review_list_ranking_and_reasons():
    df = _deals()
    out = build_review_list(df, classify_rules(df), _proba(), CLASSES, threshold=0.9)
    assert list(out["rank"]) == [1, 2, 3]
    assert out.iloc[0]["deal_id"] == "D-3"          # needs_review first
    assert out.iloc[0]["source"] == "needs_review"
    assert out.iloc[0]["second_label"] == "rule"     # dual cause stays visible
    assert (out["reason"].str.len() > 20).all()
    # priority = |diff| * (1 + (1 - confidence))
    r = out.iloc[0]
    assert abs(r["priority_score"] - abs(r["diff_amount"]) * (1 + (1 - r["confidence"]))) < 0.02


def test_reason_for_missing_deal_mentions_flagging():
    row = _deals().iloc[1]
    text = reason_for_deal(row, "data")
    assert "missing" in text and "not a" in text  # flag it, don't call it an error


def test_mapping_review_list_is_ranked():
    df = pd.DataFrame([
        {"account_id": "A-1", "account_name": "Rent Expense",
         "category": "Operating Expense", "entity": "US", "old_code": "610-US",
         "mapped_new_code": "6010-US", "mapped_account_name": "Rent Expense",
         "mapped_category": "Operating Expense", "old_balance": 100.0,
         "new_balance": 100.0, "is_mismapped": 0},
        {"account_id": "A-2", "account_name": "Rent Expense",
         "category": "Operating Expense", "entity": "UK", "old_code": "610-UK",
         "mapped_new_code": "2010-UK", "mapped_account_name": "Accounts Payable",
         "mapped_category": "Liabilities", "old_balance": 100.0,
         "new_balance": 900.0, "is_mismapped": 1},
    ])
    feats = pd.DataFrame({"collision_count": [1, 2]})
    out = build_mapping_review_list(df, np.array([0.05, 0.95]), feats)
    assert out.iloc[0]["account_id"] == "A-2"
    assert "Liabilities" in out.iloc[0]["reason"]
