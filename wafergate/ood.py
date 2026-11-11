"""Unseen-pattern scores. Higher = more likely a pattern the model never saw."""
from __future__ import annotations

import numpy as np


def msp_score(probs: np.ndarray) -> np.ndarray:
    """Baseline (Hendrycks and Gimpel 2017): 1 - max softmax probability."""
    return 1.0 - np.asarray(probs).max(axis=1)


class Mahalanobis:
    """Class-conditional Gaussians with a shared covariance on penultimate features
    (Lee et al. 2018). Score = distance to the closest class mean."""

    def fit(self, emb: np.ndarray, y_idx: np.ndarray, n_classes: int, shrink: float = 1e-3):
        emb = np.asarray(emb, dtype=np.float64)
        self.means = np.stack([emb[y_idx == k].mean(axis=0) for k in range(n_classes)])
        centered = emb - self.means[y_idx]
        cov = centered.T @ centered / len(emb)
        cov += shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self.prec = np.linalg.inv(cov)
        return self

    def distances(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float64)
        d = emb[:, None, :] - self.means[None, :, :]
        return np.einsum("nkd,de,nke->nk", d, self.prec, d)

    def score(self, emb: np.ndarray) -> np.ndarray:
        return self.distances(emb).min(axis=1)
