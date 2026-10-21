import numpy as np

from src.features import FEATURES, build_features, build_ledger_features
from src.generate_data import generate
from src.generate_ledger import generate_ledger
from src.models import (shap_feature_importance, train_diff_classifier,
                        train_mapping_detector)


def _diff_training():
    df = generate(2500, seed=21)
    d = df[(df["new_commission"] != df["old_commission"]) & (df["cause"] != "none")]
    return build_features(d), d["cause"]


def test_diff_classifier_probabilities_and_early_stopping():
    X, y = _diff_training()
    model, info = train_diff_classifier(X, y, seed=21)
    assert 1 <= info["n_estimators_used"] <= info["n_estimators_configured"]
    proba = model.predict_proba(X.head(25))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(model.classes_) == {"rounding", "rule", "timing", "data"}


def test_shap_importance_is_a_ranked_series():
    X, y = _diff_training()
    model, _ = train_diff_classifier(X, y, seed=21)
    imp = shap_feature_importance(model, X.head(300), seed=21)
    assert list(imp.index) == list(imp.sort_values(ascending=False).index)
    assert set(imp.index) == set(FEATURES)
    assert (imp >= 0).all()


def test_mapping_detector_outputs_probabilities():
    led = generate_ledger(seed=22)
    X = build_ledger_features(led)
    y = led["is_mismapped"].to_numpy()
    model = train_mapping_detector(X, y, seed=22)
    proba = model.predict_proba(X.head(20))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    # flagged-as-suspicious accounts should score above plainly correct ones
    all_proba = model.predict_proba(X)[:, 1]
    assert all_proba[y == 1].mean() > all_proba[y == 0].mean()
