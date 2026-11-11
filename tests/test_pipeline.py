"""End-to-end smoke run on a small slice of the real cache: train, register, gate, reload, monitor."""
import json
from pathlib import Path

import pytest

from wafergate import run_all

CACHE = Path(__file__).resolve().parent.parent / "data" / "wm811k_48.npz"


@pytest.mark.skipif(not CACHE.exists(), reason="data cache not present")
def test_quick_pipeline(tmp_path):
    res = run_all.main(["--quick", "--out-root", str(tmp_path)])
    out = json.loads((tmp_path / "results" / "results_quick.json").read_text())
    assert out["headline"]["candidates_evaluated"] == 3
    assert out["headline"]["audit_chain_valid"] is True
    assert out["dataset"]["lot_overlap_between_splits"] == 0
    assert len((tmp_path / "results" / "audit" / "decisions.jsonl").read_text().splitlines()) == 3
    for d in out["gate_decisions"]:
        assert d["promoted"] or d["violations"]
    assert res["headline"] == out["headline"]
