"""Batch drift monitor plus simulated tool-drift scenarios.

The monitor sees only unlabeled incoming wafers (as in production) and compares
each batch against a reference window with the Population Stability Index (PSI)
on cheap physical statistics, plus the model's own abstain rate. If either goes
past its limit, the batch is blocked and routed to engineers instead of being
auto-dispositioned.
"""
from __future__ import annotations

import numpy as np

RING_EDGES = np.linspace(0, 1.0001, 5)


def wafer_stats(X: np.ndarray) -> np.ndarray:
    """Per-wafer physical statistics: global fail rate, 4 radial ring fail rates,
    quadrant asymmetry (std of quadrant fail rates)."""
    X = np.asarray(X, dtype=np.float32)
    n, _, s, _ = X.shape
    c = (s - 1) / 2.0
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.hypot(yy - c, xx - c) / (s / 2.0)
    ang = np.arctan2(yy - c, xx - c)
    valid = X[:, 0] > 0.5
    fail = X[:, 1] * valid
    out = [fail.sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1)]
    for k in range(4):
        m = (r >= RING_EDGES[k]) & (r < RING_EDGES[k + 1])
        out.append((fail * m).sum((1, 2)) / np.maximum((valid & m).sum((1, 2)), 1))
    quads = []
    for q in range(4):
        m = (ang >= -np.pi + q * np.pi / 2) & (ang < -np.pi + (q + 1) * np.pi / 2)
        quads.append((fail * m).sum((1, 2)) / np.maximum((valid & m).sum((1, 2)), 1))
    out.append(np.std(np.stack(quads, 1), axis=1))
    return np.stack(out, 1)


STAT_NAMES = ["fail_rate", "ring0", "ring1", "ring2", "ring3", "quadrant_asym"]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(ref.min(), ref.max() + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


class DriftMonitor:
    def __init__(self, ref_stats: np.ndarray, ref_abstain_rate: float, psi_limit: float = 0.25,
                 abstain_multiplier: float = 2.0):
        self.ref = ref_stats
        self.ref_abstain = ref_abstain_rate
        self.psi_limit = psi_limit
        self.abstain_limit = max(abstain_multiplier * ref_abstain_rate, ref_abstain_rate + 0.03)

    def check(self, batch_stats: np.ndarray, batch_abstain_rate: float) -> dict:
        psis = {n: psi(self.ref[:, i], batch_stats[:, i]) for i, n in enumerate(STAT_NAMES)}
        worst = max(psis, key=psis.get)
        reasons = []
        if psis[worst] > self.psi_limit:
            reasons.append(f"PSI {worst}={psis[worst]:.3f} > {self.psi_limit}")
        if batch_abstain_rate > self.abstain_limit:
            reasons.append(f"abstain rate {batch_abstain_rate:.3f} > {self.abstain_limit:.3f}")
        return {"blocked": bool(reasons), "reasons": reasons, "max_psi": psis[worst], "max_psi_stat": worst,
                "abstain_rate": float(batch_abstain_rate)}


# ---------------------------------------------------------------- simulated tool drift
def probe_card_wear(X, rng, p=0.04):
    """Worn probe-card needles: every good die has an extra chance p of reading as a fail."""
    X = X.copy()
    valid = X[:, 0] > 0.5
    extra = (rng.random(X[:, 1].shape) < p) & valid
    X[:, 1] = np.where(extra, 1.0, X[:, 1])
    return X


def edge_bead_issue(X, rng, band=0.12, p=0.35):
    """Edge-bead removal / chuck contact problem: extra fails in the outer band on every wafer."""
    X = X.copy()
    s = X.shape[-1]
    c = (s - 1) / 2.0
    yy, xx = np.mgrid[0:s, 0:s]
    r = np.hypot(yy - c, xx - c) / (s / 2.0)
    valid = X[:, 0] > 0.5
    outer_r = np.array([r[v].max() if v.any() else 1.0 for v in valid])[:, None, None]
    band_mask = (r[None] >= outer_r - band) & valid
    extra = band_mask & (rng.random(X[:, 1].shape) < p)
    X[:, 1] = np.where(extra, 1.0, X[:, 1])
    return X


def tester_dropout(X, rng, frac=0.5):
    """Tester / data-pipeline fault: a fraction of wafers arrive with fail bits cleared in one half."""
    X = X.copy()
    hit = rng.random(len(X)) < frac
    s = X.shape[-1]
    X[hit, 1, :, s // 2 :] = 0.0
    return X


def rotate_90(X, rng):
    """Benign control: wafer orientation changed by 90 degrees (label preserving)."""
    k = rng.integers(1, 4, size=len(X))
    out = X.copy()
    for i, kk in enumerate(k):
        out[i] = np.rot90(X[i], kk, axes=(1, 2))
    return out


SCENARIOS = {
    "probe_card_wear": (probe_card_wear, True),
    "edge_bead_issue": (edge_bead_issue, True),
    "tester_half_dropout": (tester_dropout, True),
    "rotation_control": (rotate_90, False),
}
