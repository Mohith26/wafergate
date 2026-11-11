"""Handcrafted wafer-map features in the spirit of Wu, Jang and Chen (2015).

Three groups:
  * density: fail density in 13 regions (inner 3x3 grid + 4 edge quadrants) and 6 radial rings
  * radon: Radon transform of the fail map, row mean and row std, interpolated to 20 points each
  * geometry: shape of the largest connected failing region
Plus the global fail rate. Input is the (2, S, S) area-pooled tensor from data.py.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d

FAIL_THRESHOLD = 0.3
N_RADON = 20
THETA = np.arange(0.0, 180.0, 4.0)


def _polar(size: int):
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - c, xx - c) / (size / 2.0)
    ang = np.arctan2(yy - c, xx - c)
    return yy, xx, r, ang


def density_features(valid: np.ndarray, fail: np.ndarray) -> np.ndarray:
    size = valid.shape[0]
    yy, xx, r, ang = _polar(size)
    on = valid > 0.5
    feats = []
    # inner 3x3 grid restricted to r < 0.75
    inner = on & (r < 0.75)
    edges = np.linspace(0, size, 4).astype(int)
    for i in range(3):
        for j in range(3):
            m = inner & (yy >= edges[i]) & (yy < edges[i + 1]) & (xx >= edges[j]) & (xx < edges[j + 1])
            feats.append(fail[m].mean() if m.any() else 0.0)
    # 4 edge quadrants r >= 0.75
    outer = on & (r >= 0.75)
    for q in range(4):
        lo, hi = -np.pi + q * np.pi / 2, -np.pi + (q + 1) * np.pi / 2
        m = outer & (ang >= lo) & (ang < hi)
        feats.append(fail[m].mean() if m.any() else 0.0)
    # 6 radial rings
    rb = np.linspace(0, 1.0001, 7)
    for k in range(6):
        m = on & (r >= rb[k]) & (r < rb[k + 1])
        feats.append(fail[m].mean() if m.any() else 0.0)
    return np.asarray(feats, dtype=np.float32)


def radon_features(fail_bin: np.ndarray) -> np.ndarray:
    from skimage.transform import radon

    sino = radon(fail_bin.astype(float), theta=THETA, circle=False)
    out = []
    for prof in (sino.mean(axis=1), sino.std(axis=1)):
        x = np.linspace(1, len(prof), len(prof))
        f = interp1d(x, prof, kind="cubic")
        out.append(f(np.linspace(1, len(prof), N_RADON)))
    return np.concatenate(out).astype(np.float32)


def geometry_features(fail_bin: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from skimage.measure import regionprops

    size = fail_bin.shape[0]
    lab, n = ndimage.label(fail_bin, structure=np.ones((3, 3), dtype=int))  # 8-connected: diagonal scratches stay one region
    wafer_area = max(float((valid > 0.5).sum()), 1.0)
    if n == 0:
        return np.zeros(8, dtype=np.float32)
    areas = ndimage.sum(fail_bin, lab, index=np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    p = regionprops((lab == k).astype(np.uint8))[0]
    c = (size - 1) / 2.0
    cy, cx = p.centroid
    return np.asarray(
        [
            p.area / wafer_area,
            p.perimeter / size,
            p.axis_major_length / size,
            p.axis_minor_length / size,
            p.eccentricity,
            p.solidity,
            np.hypot(cy - c, cx - c) / (size / 2.0),
            n / 50.0,
        ],
        dtype=np.float32,
    )


def extract_one(x: np.ndarray) -> np.ndarray:
    valid, fail = x[0], x[1]
    fail = fail * (valid > 0.5)
    fail_bin = (fail >= FAIL_THRESHOLD).astype(np.uint8)
    gfr = np.asarray([fail[valid > 0.5].mean() if (valid > 0.5).any() else 0.0], dtype=np.float32)
    return np.concatenate([density_features(valid, fail), radon_features(fail_bin), geometry_features(fail_bin, valid), gfr])


def extract(X: np.ndarray, n_jobs: int = 8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if n_jobs <= 1 or len(X) < 2000:
        return np.stack([extract_one(x) for x in X])
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    chunks = np.array_split(X, n_jobs * 4)
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            parts = list(ex.map(_extract_chunk, chunks))
    except (PermissionError, OSError):  # sandboxes without POSIX semaphores
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            parts = list(ex.map(_extract_chunk, chunks))
    return np.concatenate(parts)


def _extract_chunk(chunk):
    return np.stack([extract_one(x) for x in chunk])


N_FEATURES = 13 + 6 + 2 * N_RADON + 8 + 1
