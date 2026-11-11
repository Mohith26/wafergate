"""Render one real wafer map per class plus the simulated drift scenarios (results/wafer_examples.png)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data as D
from . import drift as DR

ROOT = Path(__file__).resolve().parent.parent


def show(ax, x, title):
    img = np.where(x[0] > 0.5, 1.0 + x[1], 0.0)
    ax.imshow(img, cmap="viridis", vmin=0, vmax=2)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    X, y, _ = D.load_cache(ROOT / "data" / "wm811k_48.npz")
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(2, 7, figsize=(14, 4.4))
    for ax, c in zip(axes[0], ["Center", "Edge-Ring", "Edge-Loc", "Loc", "Scratch", "Random", "Donut"]):
        i = rng.choice(np.flatnonzero(y == c))
        show(ax, X[i], c + (" (held out)" if c == D.HELD_OUT_CLASS else ""))
    base = X[rng.choice(np.flatnonzero(y == "none"), 1)]
    show(axes[1][0], base[0], "none")
    show(axes[1][1], X[rng.choice(np.flatnonzero(y == "Near-full"))], "Near-full")
    for ax, (name, (fn, _)) in zip(axes[1][2:], DR.SCENARIOS.items()):
        src = base if name != "tester_half_dropout" else X[rng.choice(np.flatnonzero(y == "Edge-Ring"), 1)]
        kw = {"frac": 1.0} if name == "tester_half_dropout" else {}
        show(ax, fn(src, rng, **kw)[0], "drift: " + name.replace("_", " "))
    axes[1][6].axis("off")
    fig.tight_layout()
    out = ROOT / "results" / "wafer_examples.png"
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()
