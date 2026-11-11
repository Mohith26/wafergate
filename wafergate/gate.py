"""Promotion policy: a candidate becomes champion only if every rule passes.

The policy lives in policy.json so it can be reviewed and versioned like code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Policy:
    macro_f1_min: float = 0.70
    min_improvement_over_champion: float = 0.01
    per_class_recall_min: float = 0.50
    max_class_recall_regression: float = 0.05
    ece_max: float = 0.05
    ood_auroc_min: float = 0.75
    in_dist_abstain_max: float = 0.08

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls(**json.loads(Path(path).read_text()))


@dataclass
class Decision:
    promoted: bool
    violations: list = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def evaluate(candidate: dict, champion: dict | None, policy: Policy) -> Decision:
    """candidate/champion: {'macro_f1', 'per_class_recall': {cls: r}, 'ece', 'ood_auroc', 'in_dist_abstain_rate'}"""
    v, checks = [], {}

    def check(name, ok, detail):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            v.append(f"{name}: {detail}")

    check("macro_f1_min", candidate["macro_f1"] >= policy.macro_f1_min,
          f"{candidate['macro_f1']:.4f} vs floor {policy.macro_f1_min}")
    worst_cls = min(candidate["per_class_recall"], key=candidate["per_class_recall"].get)
    worst = candidate["per_class_recall"][worst_cls]
    check("per_class_recall_min", worst >= policy.per_class_recall_min,
          f"worst class {worst_cls} recall {worst:.4f} vs floor {policy.per_class_recall_min}")
    check("ece_max", candidate["ece"] <= policy.ece_max, f"{candidate['ece']:.4f} vs ceiling {policy.ece_max}")
    check("ood_auroc_min", candidate["ood_auroc"] >= policy.ood_auroc_min,
          f"{candidate['ood_auroc']:.4f} vs floor {policy.ood_auroc_min}")
    check("in_dist_abstain_max", candidate["in_dist_abstain_rate"] <= policy.in_dist_abstain_max,
          f"{candidate['in_dist_abstain_rate']:.4f} vs ceiling {policy.in_dist_abstain_max}")
    if champion is not None:
        need = champion["macro_f1"] + policy.min_improvement_over_champion
        check("beats_champion", candidate["macro_f1"] >= need,
              f"{candidate['macro_f1']:.4f} vs champion {champion['macro_f1']:.4f} + {policy.min_improvement_over_champion}")
        drops = {c: champion["per_class_recall"][c] - r for c, r in candidate["per_class_recall"].items()
                 if c in champion["per_class_recall"]}
        worst_drop_cls = max(drops, key=drops.get)
        check("no_class_regression", drops[worst_drop_cls] <= policy.max_class_recall_regression,
              f"largest recall drop {worst_drop_cls} {drops[worst_drop_cls]:+.4f} vs limit {policy.max_class_recall_regression}")
    return Decision(promoted=not v, violations=v, checks=checks)
