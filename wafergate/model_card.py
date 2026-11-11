"""Write MODEL_CARD.md straight from results.json so the card can never disagree with the run."""
from __future__ import annotations

from pathlib import Path


def _pct(x):
    return f"{100 * x:.1f}%"


def write_model_card(results: dict, path: str | Path) -> None:
    h = results["headline"]
    champ = h["champion"]
    ev = results["candidates"].get(champ, {}) if champ else {}
    d = results["dataset"]
    lines = [
        "# Model card: wafer-defect-classifier (current champion)",
        "",
        f"**Champion:** `{champ}`  ",
        f"**Data:** {d.get('source', 'WM-811K')}; {d['train_maps']} train / {d['val_maps']} val / {d['test_maps']} test maps, split by lot ({d['n_lots']} lots, {d['lot_overlap_between_splits']} lots shared across splits).  ",
        f"**Split hash:** `{d['split_sha256'][:16]}...`",
        "",
        "## Intended use",
        "Route electrical-wafer-sort maps to the right failure-pattern owner. Outputs one of 8 known patterns or ABSTAIN.",
        "ABSTAIN means the map looks unlike anything in training and should go to an engineer.",
        "",
        "## Not intended for",
        "Final yield disposition without a human, other fabs or products without re-validation, or die-level root cause.",
        "",
        "## Performance (lot-grouped test split, 8 known classes)",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    if ev:
        lines += [
            f"| macro-F1 | {ev['macro_f1']:.3f} |",
            f"| accuracy | {ev['accuracy']:.3f} |",
            f"| expected calibration error | {ev['ece']:.3f} (temperature {ev['temperature']:.2f}) |",
            f"| unseen-pattern (Donut) AUROC | {ev['ood_auroc']:.3f} |",
            f"| Donut wafers abstained | {ev['unseen_caught']}/{ev['unseen_total']} ({_pct(ev['abstain_recall_unseen'])}) |",
            f"| in-distribution abstain rate | {_pct(ev['in_dist_abstain_rate'])} |",
            "",
            "| class | recall | test count |",
            "|---|---|---|",
        ]
        for c, r in ev["per_class_recall"].items():
            lines.append(f"| {c} | {r:.3f} | {d['test_class_counts'][c]} |")
    lines += [
        "",
        "## Release gate",
        "Every candidate is judged against `policy.json`. Decisions this run:",
        "",
    ]
    for g in results["gate_decisions"]:
        status = "promoted" if g["promoted"] else "blocked: " + "; ".join(g["violations"])
        lines.append(f"- v{g['version']} `{g['candidate']}`: {status}")
    dr = results["drift"]
    lines += [
        "",
        "## Monitoring",
        f"Batch drift monitor (PSI limit {h['drift_psi_limit']}, abstain limit {h['drift_abstain_limit']:.3f}).",
        "",
        "| scenario | batches blocked | macro-F1 if served anyway |",
        "|---|---|---|",
    ]
    for s, v in dr.items():
        lines.append(f"| {s} | {v['blocked_batches']}/{v['batches']} | {v['macro_f1_if_served']:.3f} |")
    lines += [
        "",
        "## Known limits",
        f"- Near-full has only {d['test_class_counts'].get('Near-full', '?')} test maps, so its recall is noisy.",
        "- The `none` class was downsampled to keep training on a laptop CPU, so class priors differ from a real fab line.",
        "- Drift scenarios are simulated perturbations of real maps, not recordings of a real tool excursion.",
        "- Donut is the only held-out pattern tested; other unseen patterns may be harder or easier to catch.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")
