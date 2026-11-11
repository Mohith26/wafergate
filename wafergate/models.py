"""The two model families and the deployable wrapper around them.

Every model is wrapped in a WaferModel that exposes the same three things the
platform needs: calibrated class probabilities, an out-of-distribution score,
and an abstain threshold. The registry stores the whole wrapper, so the abstain
gate ships with the model instead of living in someone's notebook.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import extract
from .ood import Mahalanobis, msp_score

ABSTAIN = "ABSTAIN"


# ---------------------------------------------------------------- CNN
class WaferCNN(nn.Module):
    def __init__(self, n_classes: int, width: int = 32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2),
            )

        self.body = nn.Sequential(block(2, width), block(width, 2 * width), block(2 * width, 4 * width))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4 * width, n_classes)

    def embed(self, x):
        return self.pool(self.body(x)).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))


def _augment(xb: torch.Tensor) -> torch.Tensor:
    """Label-preserving wafer symmetries: random 90-degree rotations and flips."""
    k = int(torch.randint(0, 4, (1,)))
    xb = torch.rot90(xb, k, dims=(2, 3))
    if torch.rand(1) < 0.5:
        xb = torch.flip(xb, dims=(3,))
    return xb


def train_cnn(X, y_idx, n_classes, class_weight=True, epochs=10, lr=2e-3, batch=256, seed=0, log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WaferCNN(n_classes)
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_idx, dtype=np.int64))
    if class_weight:
        cnt = np.bincount(y_idx, minlength=n_classes).astype(float)
        w = torch.tensor((cnt.sum() / (n_classes * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32)
    else:
        w = None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(Xt) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i : i + batch]
            xb = _augment(Xt[idx])
            loss = F.cross_entropy(model(xb), yt[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        if log:
            log(f"epoch {ep + 1}/{epochs} loss {tot / len(Xt):.4f}")
    model.eval()
    return model


@torch.no_grad()
def cnn_forward(model, X, batch=1024):
    logits, emb = [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(np.asarray(X[i : i + batch], dtype=np.float32))
        e = model.embed(xb)
        emb.append(e.numpy())
        logits.append(model.head(e).numpy())
    return np.concatenate(logits), np.concatenate(emb)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    lt = torch.from_numpy(logits.astype(np.float64))
    yt = torch.from_numpy(np.asarray(y_idx, dtype=np.int64))

    def nll(t):
        return float(F.cross_entropy(lt / t, yt))

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- deployable wrapper
class WaferModel:
    """Calibrated classifier + OOD scorer + abstain threshold, in one object."""

    def __init__(self, kind, classes, core, temperature=1.0, ood=None, ood_kind="msp", threshold=np.inf):
        self.kind = kind
        self.classes = list(classes)
        self.core = core
        self.temperature = float(temperature)
        self.ood = ood
        self.ood_kind = ood_kind
        self.threshold = float(threshold)

    # raw pieces -------------------------------------------------------
    def _logits_emb(self, X):
        if self.kind == "cnn":
            return cnn_forward(self.core, X)
        feats = extract(X)
        p = np.clip(self.core.predict_proba(feats), 1e-7, 1.0)
        return np.log(p), feats

    def score(self, X):
        """Return (probs, ood_score)."""
        logits, emb = self._logits_emb(X)
        probs = softmax(logits / self.temperature)
        if self.ood_kind == "mahalanobis" and self.ood is not None:
            s = self.ood.score(emb)
        else:
            s = msp_score(probs)
        return probs, s

    def predict(self, X):
        probs, s = self.score(X)
        labels = np.array(self.classes, dtype=object)[probs.argmax(axis=1)]
        labels[s > self.threshold] = ABSTAIN
        return labels, probs, s


def build_gbm(F_train, y_idx, seed=0, class_weight=True):
    cw = "balanced" if class_weight else None
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                         class_weight=cw, random_state=seed, early_stopping=False)
    clf.fit(F_train, y_idx)
    return clf


def fit_threshold(scores_val_in: np.ndarray, target_in_dist_abstain: float = 0.05) -> float:
    return float(np.quantile(scores_val_in, 1.0 - target_in_dist_abstain))


def mahalanobis_from(emb, y_idx, n_classes):
    return Mahalanobis().fit(emb, y_idx, n_classes)
