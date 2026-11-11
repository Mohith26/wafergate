"""Load WM-811K, build a fixed-size tensor cache, and split it by lot.

Every wafer map in WM-811K is a variable-size grid with values
0 = no die at this position, 1 = die passed, 2 = die failed.
I convert each map to two S x S channels with area pooling:
  channel 0: fraction of the cell covered by real dies (the wafer mask)
  channel 1: fraction of those dies that failed
Area pooling (instead of nearest-neighbour resize) keeps thin scratches visible
when a 200-wide map is shrunk to 48x48.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

KNOWN_CLASSES = ["Center", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
HELD_OUT_CLASS = "Donut"  # never shown to any model: plays the role of a newly discovered pattern
ALL_LABELED = KNOWN_CLASSES + [HELD_OUT_CLASS]
SIZE = 48


def _bins(n: int, size: int):
    """Source index range [lo, hi) for each of `size` target cells. Works for both
    shrinking (several source dies per cell) and enlarging (one die feeds several cells)."""
    t = np.arange(size)
    lo = np.floor(t * n / size).astype(np.int64)
    hi = np.maximum(lo + 1, np.ceil((t + 1) * n / size).astype(np.int64))
    return lo, np.minimum(hi, n)


def resize_map(wmap: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Area-pool a ternary wafer map to (2, size, size) float32."""
    wmap = np.asarray(wmap)
    h, w = wmap.shape
    die = (wmap > 0).astype(np.float64)
    fail = (wmap == 2).astype(np.float64)

    def box_sums(a):
        integ = np.zeros((h + 1, w + 1))
        integ[1:, 1:] = a.cumsum(0).cumsum(1)
        r0, r1 = _bins(h, size)
        c0, c1 = _bins(w, size)
        return (integ[r1][:, c1] - integ[r0][:, c1] - integ[r1][:, c0] + integ[r0][:, c0])

    r0, r1 = _bins(h, size)
    c0, c1 = _bins(w, size)
    cell_n = np.outer(r1 - r0, c1 - c0).astype(np.float64)
    die_n, fail_n = box_sums(die), box_sums(fail)
    valid = die_n / cell_n
    failfrac = np.where(die_n > 0, fail_n / np.maximum(die_n, 1e-9), 0.0)
    return np.stack([valid, failfrac]).astype(np.float32)


def load_labeled(pkl_path: str | Path):
    """Return a DataFrame of the 172,950 labeled maps with clean string labels."""
    import sys

    import pandas as pd
    import pandas.core.indexes as _ci

    sys.modules.setdefault("pandas.indexes", _ci)  # very old pickles reference this path
    df = pd.read_pickle(pkl_path)
    ft = df["failureType"].map(lambda v: v if isinstance(v, str) else (str(np.ravel(v)[0]) if np.size(v) == 1 else None))
    df = df.assign(label=ft)
    return df[df["label"].isin(ALL_LABELED)].reset_index(drop=True)


def build_cache(pkl_path: str | Path, out_path: str | Path, n_none: int = 30000, seed: int = 0) -> dict:
    """Write an .npz with X (N,2,S,S float16), y (str), lot (str) and return a manifest."""
    rng = np.random.default_rng(seed)
    df = load_labeled(pkl_path)
    counts_full = df["label"].value_counts().to_dict()
    none_idx = np.flatnonzero(df["label"].values == "none")
    keep_none = rng.choice(none_idx, size=min(n_none, len(none_idx)), replace=False)
    keep = np.sort(np.concatenate([np.flatnonzero(df["label"].values != "none"), keep_none]))
    sub = df.iloc[keep].reset_index(drop=True)
    X = np.stack([resize_map(m) for m in sub["waferMap"].values]).astype(np.float16)
    y = np.array([str(v) for v in sub["label"].tolist()], dtype="U16")
    lot = np.array([str(v) for v in sub["lotName"].tolist()], dtype="U16")
    np.savez_compressed(out_path, X=X, y=y, lot=lot)
    manifest = {
        "source": "WM-811K (Wu, Jang, Chen 2015), MIR Lab public archive",
        "labeled_maps_in_source": int(len(df)),
        "class_counts_in_source": {k: int(v) for k, v in counts_full.items()},
        "none_downsampled_to": int(len(keep_none)),
        "maps_in_cache": int(len(sub)),
        "grid": SIZE,
        "seed": seed,
    }
    return manifest


def load_cache(path: str | Path):
    z = np.load(path, allow_pickle=False)
    return z["X"].astype(np.float32), z["y"], z["lot"]


def split_by_lot(lots: np.ndarray, fracs=(0.7, 0.15, 0.15), seed: int = 0):
    """Assign whole lots to train/val/test so no lot is in two splits."""
    uniq = np.unique(lots)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    a = int(round(fracs[0] * n))
    b = a + int(round(fracs[1] * n))
    groups = {"train": set(uniq[:a]), "val": set(uniq[a:b]), "test": set(uniq[b:])}
    return {k: np.flatnonzero(np.isin(lots, list(v))) for k, v in groups.items()}


def split_manifest_hash(splits: dict, y: np.ndarray, lots: np.ndarray) -> str:
    """Stable hash of which (lot, label) went where; recorded in every gate decision."""
    h = hashlib.sha256()
    for k in sorted(splits):
        idx = np.sort(splits[k])
        h.update(k.encode())
        h.update(idx.astype(np.int64).tobytes())
        h.update(json.dumps(sorted(set(lots[idx].tolist()))[:50]).encode())
    return h.hexdigest()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", default="data/wm811k_48.npz")
    ap.add_argument("--n-none", type=int, default=30000)
    a = ap.parse_args()
    man = build_cache(a.pkl, a.out, a.n_none)
    Path(a.out).with_suffix(".manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
