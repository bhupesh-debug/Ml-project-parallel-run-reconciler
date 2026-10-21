"""Model training for Problems A and B, plus SHAP explanations.

Both models are gradient boosting, per the brief: shallow trees, early
stopping on an internal holdout of the training data, no tuning on the test
set. Small tabular data does not need more than this.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance


def train_diff_classifier(X_train: pd.DataFrame, y_train, seed: int = 42):
    """Problem A: multi-class cause classifier with early stopping.

    Returns (model, info) where info records how many trees early stopping
    actually used out of the configured maximum.
    """
    # scikit-learn >= 1.8: setting n_iter_no_change (with validation_fraction)
    # switches on early stopping for GradientBoostingClassifier.
    model = GradientBoostingClassifier(
        n_estimators=400, max_depth=2, learning_rate=0.06, subsample=0.9,
        validation_fraction=0.15, n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    info = {"n_estimators_configured": 400,
            "n_estimators_used": int(model.n_estimators_)}
    return model, info


def train_mapping_detector(X_train: pd.DataFrame, y_train, seed: int = 42):
    """Problem B: binary mis-mapping detector. Balanced sample weights handle
    the ~10% positive rate (gradient boosting has no class_weight knob)."""
    y = np.asarray(y_train)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    weights = np.where(y == 1, n_neg / max(n_pos, 1), 1.0)
    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=2, learning_rate=0.08, subsample=0.9,
        random_state=seed,
    )
    model.fit(X_train, y, sample_weight=weights)
    return model


def shap_feature_importance(model, X: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Mean |SHAP value| per feature, aggregated across classes.

    Falls back to permutation importance if SHAP is unavailable or errors —
    callers should treat the returned series as 'importance', and the
    pipeline records which method produced it.
    """
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = explainer(X).values
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:      # (n, features, classes)
            imp = np.abs(arr).mean(axis=(0, 2))
        elif arr.ndim == 2:    # (n, features)
            imp = np.abs(arr).mean(axis=0)
        else:
            raise ValueError(f"unexpected SHAP shape {arr.shape}")
        return pd.Series(imp, index=X.columns).sort_values(ascending=False)
    except Exception:
        result = permutation_importance(model, X, model.predict(X),
                                        n_repeats=5, random_state=seed)
        return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
