"""Small, dependency-light metric helpers so the gate logic is easy to audit."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_auc_score


def macro_f1(y_true, y_pred, labels) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_recall(y_true, y_pred, labels) -> dict:
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {lab: float(v) for lab, v in zip(labels, r)}


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, n_bins: int = 15) -> float:
    """Standard top-label ECE with equal-width confidence bins."""
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_idx)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def auroc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """AUROC for 'higher score = more likely out-of-distribution'."""
    y = np.r_[np.zeros(len(scores_in)), np.ones(len(scores_out))]
    return float(roc_auc_score(y, np.r_[scores_in, scores_out]))


def abstain_stats(scores_in, scores_out, threshold) -> dict:
    """Abstain when score > threshold. Positive class = the unseen pattern."""
    tp = int((np.asarray(scores_out) > threshold).sum())
    fn = int(len(scores_out) - tp)
    fp = int((np.asarray(scores_in) > threshold).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "abstain_recall_unseen": float(recall),
        "abstain_precision_unseen": float(precision),
        "in_dist_abstain_rate": float(fp / max(len(scores_in), 1)),
        "unseen_caught": tp,
        "unseen_total": int(len(scores_out)),
        "in_dist_abstained": fp,
        "in_dist_total": int(len(scores_in)),
    }
