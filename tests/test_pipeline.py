import numpy as np

from src.generate_data import CAUSES, generate
from src.features import FEATURES, build_features
from src.rules import classify_rules
from src.run_pipeline import run


def test_generate_is_reproducible():
    a, b = generate(500, seed=1), generate(500, seed=1)
    assert a.equals(b)
    assert set(a["cause"]).issubset(set(CAUSES))


def test_features_shape_and_no_nans():
    df = generate(500, seed=2)
    X = build_features(df)
    assert list(X.columns) == FEATURES
    assert not X.isna().any().any()


def test_rules_return_one_label_per_row():
    df = generate(500, seed=3)
    out = classify_rules(df)
    assert len(out) == len(df)
    assert set(out).issubset(set(CAUSES))


def test_exact_matches_are_none():
    df = generate(500, seed=4)
    same = df["new_commission"] == df["old_commission"]
    assert (classify_rules(df)[same.to_numpy()] == "none").all()


def test_pipeline_runs(tmp_path):
    r = run(n_deals=1500, seed=5, out_dir=str(tmp_path),
            data_dir=str(tmp_path / "data"))
    assert 0.0 <= r["gradient_boosting"]["macro_f1"] <= 1.0
    assert (tmp_path / "metrics.json").exists()


def test_pipeline_writes_all_artifacts(tmp_path):
    r = run(n_deals=1500, seed=5, out_dir=str(tmp_path),
            data_dir=str(tmp_path / "data"))
    for name in ("metrics.json", "model_report.md", "review_list.csv",
                 "mapping_review_list.csv", "confusion_matrix.png",
                 "feature_importance.png", "shap_importance.png",
                 "waterfall.png", "feed_health_trend.png"):
        assert (tmp_path / name).exists(), name
    for name in ("deals.csv", "ledger.csv", "feed_health.csv"):
        assert (tmp_path / "data" / name).exists(), name
    # All three problems are reported.
    assert r["problem_b"]["n_accounts"] > 0
    assert r["problem_c"]["n_months"] == 24
    # Cascade accounts for every test difference exactly once.
    assert sum(r["cascade_sources"].values()) == r["n_test_differences"]
    assert 0.0 <= r["problem_b"]["metrics"]["roc_auc"] <= 1.0
