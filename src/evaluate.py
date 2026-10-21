"""Evaluation helpers for Problems A and B."""
import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

from .generate_data import DIFF_CAUSES


def summarize(y_true, y_pred, label):
    rep = classification_report(y_true, y_pred, labels=DIFF_CAUSES, output_dict=True, zero_division=0)
    return {
        "model": label,
        "accuracy": float(rep["accuracy"]),
        "macro_f1": float(f1_score(y_true, y_pred, labels=DIFF_CAUSES, average="macro", zero_division=0)),
        "per_class": {c: {k: float(rep[c][k]) for k in ("precision", "recall", "f1-score", "support")} for c in DIFF_CAUSES},
    }


def confidence_coverage(proba, classes, y_true, threshold=0.9):
    """Share of items auto-classified at a confidence threshold, and accuracy among them."""
    conf = proba.max(axis=1)
    pred = np.array(classes)[proba.argmax(axis=1)]
    mask = conf >= threshold
    share = float(mask.mean())
    acc = float((pred[mask] == np.asarray(y_true)[mask]).mean()) if mask.any() else float("nan")
    return share, acc


def confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=DIFF_CAUSES)


def top_k_labels(proba, classes, k=2):
    """Top-k predicted labels per row, best first."""
    order = np.argsort(-proba, axis=1)[:, :k]
    cls = np.array(classes)
    return cls[order]


def top2_accuracy(proba, classes, y_true):
    """Share of rows whose true label is in the model's top two."""
    top2 = top_k_labels(proba, classes, k=2)
    return float(np.mean([t in row for t, row in zip(np.asarray(y_true), top2)]))


def dual_cause_recall(proba, classes, primary, secondary):
    """For dual-cause rows: share where BOTH injected causes appear in the
    model's top two predictions (order-insensitive)."""
    if len(primary) == 0:
        return float("nan")
    top2 = top_k_labels(proba, classes, k=2)
    hits = [set(row) >= {p, s} for row, p, s in zip(top2, primary, secondary)]
    return float(np.mean(hits))


def summarize_binary(y_true, y_score, ks=(10, 20)):
    """Binary-detector metrics plus precision/recall at review-list depths."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    out = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "n_test": int(len(y_true)),
        "n_positive_test": int(y_true.sum()),
    }
    for k in ks:
        top = order[:k]
        hits = int(y_true[top].sum())
        out[f"precision_at_{k}"] = hits / k
        out[f"recall_at_{k}"] = hits / max(int(y_true.sum()), 1)
    return out
