"""Reproduce every number in results/results.json.

    python -m wafergate.run_all            # full run (about 10 minutes on a laptop CPU)

Pipeline: lot-grouped split -> three candidates trained and logged to MLflow ->
each candidate registered and judged by the promotion policy -> champion reloaded
from the registry and re-scored -> drift monitor run over clean and drifted
batches -> results.json, audit log, model card.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from . import drift as DR
from . import features as FE
from . import metrics as M
from . import models as MD
from . import registry as R
from .gate import Policy, evaluate
from .model_card import write_model_card

ROOT = Path(__file__).resolve().parent.parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_model(wm: MD.WaferModel, X_te, y_te_idx, X_unseen, X_val, classes):
    """Metrics the gate consumes. Classification metrics use the argmax (no abstain);
    abstain metrics are reported separately."""
    probs_te, s_te = wm.score(X_te)
    _, s_un = wm.score(X_unseen)
    pred = probs_te.argmax(1)
    y_names = np.array(classes)[y_te_idx]
    p_names = np.array(classes)[pred]
    out = {
        "macro_f1": M.macro_f1(y_names, p_names, classes),
        "accuracy": float((pred == y_te_idx).mean()),
        "per_class_recall": M.per_class_recall(y_names, p_names, classes),
        "ece": M.expected_calibration_error(probs_te, y_te_idx),
        "ood_auroc": M.auroc(s_te, s_un),
        "temperature": wm.temperature,
        "ood_method": wm.ood_kind,
    }
    ab = M.abstain_stats(s_te, s_un, wm.threshold)
    out.update(ab)
    kept = s_te <= wm.threshold
    out["macro_f1_on_kept"] = M.macro_f1(y_names[kept], p_names[kept], classes)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "data" / "wm811k_48.npz"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run for CI")
    ap.add_argument("--out-root", default=str(ROOT), help="where results/, mlruns/ and MODEL_CARD.md go")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    t0 = time.time()

    out_root = Path(a.out_root)
    res_dir = out_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    mlruns = out_root / "mlruns"
    if mlruns.exists():
        shutil.rmtree(mlruns)
    audit_path = res_dir / "audit" / "decisions.jsonl"
    if audit_path.exists():
        audit_path.unlink()

    # ------------------------------------------------------------ data
    X, y, lot = D.load_cache(a.cache)
    splits = D.split_by_lot(lot)
    classes = D.KNOWN_CLASSES
    cidx = {c: i for i, c in enumerate(classes)}
    unseen_mask = y == D.HELD_OUT_CLASS

    def known(idx):
        idx = idx[~unseen_mask[idx]]
        return idx

    tr, va, te = known(splits["train"]), known(splits["val"]), known(splits["test"])
    un = np.flatnonzero(unseen_mask)
    if a.quick:
        rng = np.random.default_rng(0)
        tr, va, te = (np.sort(rng.choice(i, min(len(i), n), replace=False)) for i, n in ((tr, 3000), (va, 1000), (te, 1000)))
        un = un[:100]
        a.epochs = 1
    ytr, yva, yte = (np.array([cidx[v] for v in y[i]]) for i in (tr, va, te))
    split_hash = D.split_manifest_hash({"train": tr, "val": va, "test": te, "unseen": un}, y, lot)
    data_info = json.loads((Path(a.cache).with_suffix(".manifest.json")).read_text()) if Path(a.cache).with_suffix(".manifest.json").exists() else {}
    data_info.update({
        "split": "grouped by lot, 70/15/15 of lots",
        "n_lots": int(len(np.unique(lot))),
        "train_maps": int(len(tr)), "val_maps": int(len(va)), "test_maps": int(len(te)),
        "unseen_pattern": D.HELD_OUT_CLASS, "unseen_maps": int(len(un)),
        "lot_overlap_between_splits": int(len(set(lot[tr]) & set(lot[te])) + len(set(lot[tr]) & set(lot[va])) + len(set(lot[va]) & set(lot[te]))),
        "test_class_counts": {c: int((yte == i).sum()) for i, c in enumerate(classes)},
        "split_sha256": split_hash,
    })
    log(f"data: train {len(tr)} val {len(va)} test {len(te)} unseen {len(un)}")

    # ------------------------------------------------------------ candidates
    log("extracting handcrafted features")
    Ftr, Fva = FE.extract(X[tr]), FE.extract(X[va])
    candidates = []

    log("candidate 1: handcrafted features + gradient boosting")
    gbm = MD.build_gbm(Ftr, ytr)
    lv = np.log(np.clip(gbm.predict_proba(Fva), 1e-7, 1))
    wm_gbm = MD.WaferModel("gbm", classes, gbm, temperature=MD.fit_temperature(lv, yva), ood_kind="msp")
    _, s_val = wm_gbm.score(X[va])
    wm_gbm.threshold = MD.fit_threshold(s_val)
    candidates.append(("v1_gbm_handcrafted", wm_gbm, {"model": "HistGradientBoosting", "features": FE.N_FEATURES, "class_weight": "balanced", "ood": "msp"}))

    def cnn_candidate(name, class_weight, epochs):
        log(f"candidate {name}: CNN class_weight={class_weight} epochs={epochs}")
        net = MD.train_cnn(X[tr], ytr, len(classes), class_weight=class_weight, epochs=epochs, log=log)
        lv, ev = MD.cnn_forward(net, X[va])
        maha = MD.mahalanobis_from(MD.cnn_forward(net, X[tr])[1], ytr, len(classes))
        wm = MD.WaferModel("cnn", classes, net, temperature=MD.fit_temperature(lv, yva), ood=maha, ood_kind="mahalanobis")
        wm.threshold = MD.fit_threshold(maha.score(ev))
        return wm

    wm_rushed = cnn_candidate("v2_cnn_rushed", class_weight=False, epochs=max(1, a.epochs // 4))
    candidates.append(("v2_cnn_rushed_retrain", wm_rushed, {"model": "WaferCNN", "class_weight": "none", "epochs": max(1, a.epochs // 4), "ood": "mahalanobis"}))
    wm_cnn = cnn_candidate("v3_cnn", class_weight=True, epochs=a.epochs)
    candidates.append(("v3_cnn_weighted", wm_cnn, {"model": "WaferCNN", "class_weight": "sqrt_inverse_freq", "epochs": a.epochs, "ood": "mahalanobis"}))

    # ------------------------------------------------------------ registry + gate
    client = R.setup(mlruns)
    policy = Policy.load(ROOT / "policy.json")
    audit = R.AuditLog(audit_path)
    evals, decisions = {}, []
    champion_metrics = None
    for name, wm, params in candidates:
        log(f"evaluating {name}")
        ev = evaluate_model(wm, X[te], yte, X[un], X[va], classes)
        evals[name] = ev
        run_id, version = R.log_and_register(client, wm, name, params, ev, mlruns / "staging", str(ROOT / "wafergate"))
        dec = evaluate(ev, champion_metrics, policy)
        R.apply_decision(client, version, dec)
        rec = audit.append({"candidate": name, "model_version": version, "run_id": run_id,
                            "model_sha256": R.model_sha256(wm), "split_sha256": split_hash,
                            "promoted": dec.promoted, "violations": dec.violations,
                            "champion_before": R.current_champion(client) if not dec.promoted else None})
        decisions.append({"candidate": name, "version": version, "promoted": dec.promoted,
                          "violations": dec.violations, "audit_hash": rec["hash"]})
        log(f"  -> {'PROMOTED' if dec.promoted else 'BLOCKED'} {dec.violations}")
        if dec.promoted:
            champion_metrics = ev

    champ_version = R.current_champion(client)
    champ_name = next(d["candidate"] for d in decisions if d["version"] == champ_version) if champ_version else None
    champ_wm = dict((n, w) for n, w, _ in candidates)[champ_name] if champ_name else wm_cnn

    # reload the champion through the registry and make sure it serves the same answers
    served = R.load_champion() if champ_version else None
    if served is not None:
        served_pred = np.array(served.predict(X[te]))
        mem_pred, _, _ = champ_wm.predict(X[te])
        registry_match = float((served_pred == mem_pred).mean())
    else:
        registry_match = None

    # MSP baseline for the abstain comparison, on the same CNN
    probs_te, _ = wm_cnn.score(X[te])
    probs_un, _ = wm_cnn.score(X[un])
    probs_va, _ = wm_cnn.score(X[va])
    from .ood import msp_score
    msp_thr = MD.fit_threshold(msp_score(probs_va))
    msp_block = {"ood_auroc": M.auroc(msp_score(probs_te), msp_score(probs_un)),
                 **M.abstain_stats(msp_score(probs_te), msp_score(probs_un), msp_thr)}

    # ------------------------------------------------------------ drift monitor
    log("drift monitor")
    rng = np.random.default_rng(7)
    ref_stats = DR.wafer_stats(X[va])
    _, s_ref = champ_wm.score(X[va])
    monitor = DR.DriftMonitor(ref_stats, float((s_ref > champ_wm.threshold).mean()))
    n_batches, bsz = (4, 200) if a.quick else (20, 400)
    drift_out = {}
    for scen in ["clean"] + list(DR.SCENARIOS):
        blocks, yt_all, yp_all, reasons = [], [], [], []
        for b in range(n_batches):
            idx = rng.choice(te, size=bsz, replace=False)
            xb = X[idx]
            if scen != "clean":
                xb = DR.SCENARIOS[scen][0](xb, rng)
            labels, probs, s = champ_wm.predict(xb)
            chk = monitor.check(DR.wafer_stats(xb), float((s > champ_wm.threshold).mean()))
            blocks.append(chk["blocked"])
            reasons.append(chk["reasons"][0] if chk["reasons"] else "")
            yt_all.append(y[idx])
            yp_all.append(np.array(classes)[probs.argmax(1)])
        yt_all, yp_all = np.concatenate(yt_all), np.concatenate(yp_all)
        drift_out[scen] = {
            "harmful": bool(DR.SCENARIOS[scen][1]) if scen != "clean" else False,
            "batches": n_batches, "batch_size": bsz,
            "blocked_batches": int(sum(blocks)),
            "block_rate": float(np.mean(blocks)),
            "macro_f1_if_served": M.macro_f1(yt_all, yp_all, classes),
            "example_reason": next((r for r in reasons if r), ""),
        }
        log(f"  {scen}: blocked {sum(blocks)}/{n_batches} macroF1 {drift_out[scen]['macro_f1_if_served']:.3f}")
    harmful = [s for s in DR.SCENARIOS if DR.SCENARIOS[s][1]]
    drift_summary = {
        "harmful_batches_blocked": int(sum(drift_out[s]["blocked_batches"] for s in harmful)),
        "harmful_batches_total": int(sum(drift_out[s]["batches"] for s in harmful)),
        "clean_false_alarms": drift_out["clean"]["blocked_batches"],
        "clean_batches": drift_out["clean"]["batches"],
        "benign_rotation_blocked": drift_out["rotation_control"]["blocked_batches"],
        "clean_macro_f1": drift_out["clean"]["macro_f1_if_served"],
        "worst_harmful_macro_f1": min(drift_out[s]["macro_f1_if_served"] for s in harmful),
        "psi_limit": monitor.psi_limit, "abstain_limit": monitor.abstain_limit,
    }

    # ------------------------------------------------------------ headline + write
    base, cnn = evals["v1_gbm_handcrafted"], evals["v3_cnn_weighted"]
    rare = ["Near-full", "Scratch", "Random"]
    headline = {
        "baseline_macro_f1": base["macro_f1"],
        "cnn_macro_f1": cnn["macro_f1"],
        "macro_f1_gain_points": 100 * (cnn["macro_f1"] - base["macro_f1"]),
        "baseline_rare_class_recall": {c: base["per_class_recall"][c] for c in rare},
        "cnn_rare_class_recall": {c: cnn["per_class_recall"][c] for c in rare},
        "cnn_ece": cnn["ece"], "baseline_ece": base["ece"],
        "unseen_donut_auroc_mahalanobis": cnn["ood_auroc"],
        "unseen_donut_auroc_msp_baseline": msp_block["ood_auroc"],
        "unseen_donut_abstain_recall_mahalanobis": cnn["abstain_recall_unseen"],
        "unseen_donut_abstain_recall_msp_baseline": msp_block["abstain_recall_unseen"],
        "in_dist_abstain_rate_mahalanobis": cnn["in_dist_abstain_rate"],
        "candidates_evaluated": len(decisions),
        "candidates_promoted": int(sum(d["promoted"] for d in decisions)),
        "candidates_blocked": int(sum(not d["promoted"] for d in decisions)),
        "champion": champ_name,
        "registry_reload_prediction_match": registry_match,
        "audit_chain_valid": audit.verify(),
        **{f"drift_{k}": v for k, v in drift_summary.items()},
    }
    results = {
        "dataset": data_info,
        "policy": json.loads((ROOT / "policy.json").read_text()),
        "candidates": evals,
        "gate_decisions": decisions,
        "msp_baseline_on_cnn": msp_block,
        "drift": drift_out,
        "headline": headline,
        "runtime_seconds": round(time.time() - t0, 1),
        "quick_mode": bool(a.quick),
    }
    out = res_dir / ("results_quick.json" if a.quick else "results.json")
    out.write_text(json.dumps(results, indent=2))
    if not a.quick:
        write_model_card(results, out_root / "MODEL_CARD.md")
    log(f"wrote {out} in {results['runtime_seconds']}s")
    print(json.dumps(headline, indent=2))
    return results


if __name__ == "__main__":
    main()
