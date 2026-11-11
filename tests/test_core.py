import json

import numpy as np
import pytest

from wafergate import data as D
from wafergate import drift as DR
from wafergate import features as FE
from wafergate import metrics as M
from wafergate.gate import Policy, evaluate
from wafergate.ood import Mahalanobis, msp_score
from wafergate.registry import AuditLog

S = D.SIZE


def disk_map(n=60, pattern=None, rng=None):
    """Build a raw ternary wafer map (0 off-wafer, 1 pass, 2 fail) with a chosen fail pattern."""
    rng = rng or np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n]
    c = (n - 1) / 2
    r = np.hypot(yy - c, xx - c) / (n / 2)
    m = np.where(r <= 1.0, 1, 0)
    if pattern == "center":
        m[(r < 0.3)] = 2
    elif pattern == "ring":
        m[(r > 0.85) & (r <= 1.0)] = 2
    elif pattern == "scratch":
        for t in range(10, 50):
            m[t, t] = 2
    elif pattern == "random":
        m[(rng.random((n, n)) < 0.3) & (r <= 1.0)] = 2
    return m


def to_x(raw):
    return D.resize_map(raw)


# ------------------------------------------------------------------ data
def test_resize_shape_and_range():
    x = to_x(disk_map(77, "center"))
    assert x.shape == (2, S, S)
    assert x.min() >= 0 and x.max() <= 1


def test_resize_preserves_fail_fraction():
    raw = disk_map(96, "random")
    x = to_x(raw)
    true = (raw == 2).sum() / (raw > 0).sum()
    # area pooling: die-weighted average of cell fail fractions equals the true fail rate
    die_counts = x[0]
    est = (x[1] * die_counts).sum() / die_counts.sum()
    assert abs(est - true) < 0.02


def test_resize_keeps_thin_scratch_visible():
    raw = disk_map(200)
    for t in range(40, 160):
        raw[t, t] = 2
    x = to_x(raw)
    assert (x[1] > 0).sum() >= 25  # a one-die-wide line survives a 200 -> 48 shrink


def test_split_by_lot_has_no_overlap_and_covers_all():
    lots = np.array([f"lot{i // 7}" for i in range(700)])
    s = D.split_by_lot(lots, seed=3)
    sets = {k: set(lots[v]) for k, v in s.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert sum(len(v) for v in s.values()) == 700


def test_split_hash_is_deterministic_and_sensitive():
    lots = np.array([f"lot{i // 5}" for i in range(100)])
    y = np.array(["none"] * 100)
    s1 = D.split_by_lot(lots, seed=0)
    s2 = D.split_by_lot(lots, seed=0)
    s3 = D.split_by_lot(lots, seed=1)
    assert D.split_manifest_hash(s1, y, lots) == D.split_manifest_hash(s2, y, lots)
    assert D.split_manifest_hash(s1, y, lots) != D.split_manifest_hash(s3, y, lots)


def test_held_out_class_not_in_known():
    assert D.HELD_OUT_CLASS not in D.KNOWN_CLASSES


# ------------------------------------------------------------------ features
def test_feature_length():
    f = FE.extract_one(to_x(disk_map(60, "center")))
    assert f.shape == (FE.N_FEATURES,)
    assert np.all(np.isfinite(f))


def test_center_vs_ring_density():
    fc = FE.extract_one(to_x(disk_map(60, "center")))
    fr = FE.extract_one(to_x(disk_map(60, "ring")))
    inner_center = fc[4]  # middle cell of the inner 3x3 grid
    outer_center = fc[9:13].mean()  # 4 edge quadrants
    assert inner_center > 0.5 and outer_center < 0.1
    assert fr[9:13].mean() > 0.5 and fr[4] < 0.1


def test_geometry_scratch_is_elongated():
    f = FE.extract_one(to_x(disk_map(60, "scratch")))
    geo = f[13 + 6 + 2 * FE.N_RADON : 13 + 6 + 2 * FE.N_RADON + 8]
    assert geo[4] > 0.95  # eccentricity of the largest region


def test_clean_wafer_has_zero_geometry():
    f = FE.extract_one(to_x(disk_map(60)))
    assert f[-1] == 0.0


# ------------------------------------------------------------------ metrics
def test_ece_perfect_and_overconfident():
    y = np.array([0, 1, 0, 1])
    perfect = np.eye(2)[y]
    assert M.expected_calibration_error(perfect, y) == pytest.approx(0.0)
    wrong = np.eye(2)[1 - y]
    assert M.expected_calibration_error(wrong, y) == pytest.approx(1.0)


def test_auroc_and_abstain_stats():
    s_in = np.array([0.1, 0.2, 0.3, 0.4])
    s_out = np.array([0.9, 0.8, 0.35])
    assert M.auroc(s_in, s_out) == pytest.approx(11 / 12)
    st = M.abstain_stats(s_in, s_out, threshold=0.32)
    assert st["unseen_caught"] == 3 and st["in_dist_abstained"] == 1
    assert st["abstain_precision_unseen"] == pytest.approx(0.75)


def test_macro_f1_and_recall():
    y = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    assert M.per_class_recall(y, p, ["a", "b"]) == {"a": 0.5, "b": 1.0}
    assert 0 < M.macro_f1(y, p, ["a", "b"]) < 1


# ------------------------------------------------------------------ OOD
def test_mahalanobis_flags_far_points():
    rng = np.random.default_rng(0)
    emb = np.r_[rng.normal(0, 1, (200, 4)), rng.normal(5, 1, (200, 4))]
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    m = Mahalanobis().fit(emb, y, 2)
    near = m.score(rng.normal(0, 1, (50, 4)))
    far = m.score(rng.normal(-10, 1, (50, 4)))
    assert far.min() > near.max()


def test_msp_score():
    assert np.allclose(msp_score(np.array([[0.9, 0.1], [0.5, 0.5]])), [0.1, 0.5])


# ------------------------------------------------------------------ gate
def cand(f1=0.8, worst=0.7, ece=0.02, auroc=0.9, abst=0.05):
    return {"macro_f1": f1, "per_class_recall": {"a": 0.9, "b": worst}, "ece": ece, "ood_auroc": auroc,
            "in_dist_abstain_rate": abst}


def test_gate_promotes_clean_first_candidate():
    d = evaluate(cand(), None, Policy())
    assert d.promoted and d.violations == []


@pytest.mark.parametrize("kw,rule", [({"f1": 0.5}, "macro_f1_min"), ({"worst": 0.3}, "per_class_recall_min"),
                                     ({"ece": 0.2}, "ece_max"), ({"auroc": 0.6}, "ood_auroc_min"),
                                     ({"abst": 0.2}, "in_dist_abstain_max")])
def test_gate_blocks_each_floor(kw, rule):
    d = evaluate(cand(**kw), None, Policy())
    assert not d.promoted
    assert any(v.startswith(rule) for v in d.violations)


def test_gate_requires_improvement_over_champion():
    champ = cand(f1=0.80)
    d = evaluate(cand(f1=0.805), champ, Policy())
    assert not d.promoted and any(v.startswith("beats_champion") for v in d.violations)


def test_gate_blocks_rare_class_regression_even_when_macro_f1_improves():
    champ = cand(f1=0.80, worst=0.80)
    d = evaluate(cand(f1=0.85, worst=0.70), champ, Policy())
    assert not d.promoted and any(v.startswith("no_class_regression") for v in d.violations)


def test_policy_file_loads(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"macro_f1_min": 0.9}))
    assert Policy.load(p).macro_f1_min == 0.9


# ------------------------------------------------------------------ audit log
def test_audit_chain_verifies_and_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    log.append({"candidate": "v1", "promoted": True})
    log.append({"candidate": "v2", "promoted": False})
    assert log.verify()
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    rec = json.loads(lines[1])
    rec["promoted"] = True  # someone flips a blocked decision
    lines[1] = json.dumps(rec, sort_keys=True)
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    assert not log.verify()


# ------------------------------------------------------------------ drift
def batch(pattern_list, rng):
    return np.stack([to_x(disk_map(60, p, rng)) for p in pattern_list])


def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert DR.psi(a, a) < 1e-6
    assert DR.psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_probe_card_wear_raises_fail_rate():
    rng = np.random.default_rng(0)
    X = batch([None] * 20, rng)
    Y = DR.probe_card_wear(X, rng, p=0.1)
    assert DR.wafer_stats(Y)[:, 0].mean() > DR.wafer_stats(X)[:, 0].mean() + 0.05


def test_edge_bead_hits_outer_ring_only():
    rng = np.random.default_rng(0)
    X = batch([None] * 10, rng)
    st0, st1 = DR.wafer_stats(X), DR.wafer_stats(DR.edge_bead_issue(X, rng))
    assert st1[:, 4].mean() > st0[:, 4].mean() + 0.1  # outer ring
    assert st1[:, 1].mean() == pytest.approx(st0[:, 1].mean())  # center untouched


def test_tester_dropout_clears_half():
    rng = np.random.default_rng(0)
    X = batch(["random"] * 10, rng)
    Y = DR.tester_dropout(X, rng, frac=1.0)
    assert Y[:, 1, :, S // 2 :].sum() == 0 and Y[:, 1, :, : S // 2].sum() > 0


def test_rotation_is_label_preserving_for_radial_stats():
    rng = np.random.default_rng(0)
    X = batch(["center", "ring", "random"] * 5, rng)
    st0, st1 = DR.wafer_stats(X), DR.wafer_stats(DR.rotate_90(X, rng))
    assert np.allclose(st0[:, :5], st1[:, :5], atol=0.03)


def test_monitor_quiet_on_reference_and_blocks_shift():
    rng = np.random.default_rng(1)
    pats = [None, "center", "ring", "scratch"] * 50
    ref = DR.wafer_stats(batch(pats, rng))
    mon = DR.DriftMonitor(ref, ref_abstain_rate=0.05)
    same = mon.check(DR.wafer_stats(batch(pats, rng)), 0.05)
    assert not same["blocked"]
    worn = mon.check(DR.wafer_stats(DR.probe_card_wear(batch(pats, rng), rng, p=0.08)), 0.05)
    assert worn["blocked"]
    abst = mon.check(DR.wafer_stats(batch(pats, rng)), 0.30)
    assert abst["blocked"] and any("abstain" in r for r in abst["reasons"])


def test_resize_small_map_has_no_holes():
    # many WM-811K maps are smaller than the 48x48 grid; enlarging must not leave empty rows/columns
    raw = disk_map(25)
    x = to_x(raw)
    yy, xx = np.mgrid[0:S, 0:S]
    c = (S - 1) / 2
    inside = np.hypot(yy - c, xx - c) < 0.8 * S / 2
    assert (x[0][inside] > 0.5).all()


def test_resize_identity_when_same_size():
    raw = disk_map(S, "center")
    x = to_x(raw)
    assert np.array_equal(x[0] > 0.5, raw > 0)
    assert np.array_equal(x[1] > 0.5, raw == 2)
