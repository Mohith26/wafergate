# WaferGate: a governed release pipeline for a wafer-map defect classifier

## Problem
Fabs run electrical wafer sort on every wafer and get a die-level pass/fail map. The *spatial pattern*
of the failing dies (a ring at the edge, a blob in the center, a scratch) points to a specific
process or tool problem, so classifying the pattern quickly routes the wafer to the right engineer.
Training a classifier for this is a solved-ish research problem. Shipping one is not:

- the class distribution is extremely skewed (Near-full and Donut are a fraction of a percent of maps),
  so a model that "improves accuracy" can quietly lose a rare class;
- fabs keep discovering failure modes nobody labeled, and a softmax classifier will confidently
  call a brand-new pattern one of the old ones;
- tools drift (probe cards wear, recipes change), and a model that was fine at release silently degrades.

## Why it matters to the target team
ASML's CDO AI Foundation & Platform team builds reusable platforms and governance frameworks so teams
across engineering and manufacturing can "safely deploy AI at scale" and "move from experimentation to
operational deployment". WaferGate is a small, end-to-end version of exactly that: experiment tracking,
a model registry, a written promotion policy that a candidate must pass before it becomes the
champion, an abstain path for unseen patterns, a drift monitor that blocks serving, an append-only
audit log, and an auto-generated model card. All of it on a real semiconductor manufacturing dataset.

## Data
WM-811K (Wu, Jang and Chen, IEEE Trans. Semiconductor Manufacturing 2015): 811,457 real wafer maps from
46,393 lots, 172,950 labeled with 9 classes (Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random,
Scratch, none). Downloaded from the MIR Lab public archive (archived copy of the official
`mirlab.org/dataSet/public/MIR-WM811K.zip`, no login).

- Splits are grouped **by lot** (no lot appears in two splits), so wafers from the same lot cannot leak
  between train and test.
- The `none` class is downsampled (documented in results.json) so feature extraction fits on a laptop CPU.
- **Donut is held out entirely from training** and plays the role of a newly discovered failure mode.

## Design
1. `data.py`: load the pickle, normalize labels, lot-grouped train/val/test split, resize every map
   to 32x32 with fail-fraction pooling.
2. `features.py`: the Wu et al. style handcrafted features: 13-region fail density, Radon transform
   row-mean/row-std profiles, and geometry of the largest failing connected region.
3. Models: (a) baseline: handcrafted features + HistGradientBoosting; (b) small CNN (3 conv blocks,
   class-weighted loss) with temperature scaling fit on validation.
4. `ood.py`: abstain gate using Mahalanobis distance on CNN penultimate features (class-conditional
   Gaussians with tied covariance), threshold set on validation for a 5% in-distribution abstain rate.
5. `registry.py` + `gate.py`: MLflow (local sqlite) tracking + model registry. Every candidate is
   logged, registered, and evaluated against a policy file (`policy.json`): macro-F1 floor and a
   minimum improvement over the current champion, a per-class recall floor, a max per-class recall
   regression vs champion, an ECE ceiling, an OOD AUROC floor. Passing candidates get the `champion`
   alias; failing ones are tagged `blocked` with the violated rules. Every decision is appended to
   `audit/decisions.jsonl` with a hash of the data split manifest and the model.
6. `drift.py`: a monitor over batches of incoming wafers that compares the batch to the reference
   (validation) window with PSI on input statistics plus the abstain rate. Simulated tool drift
   scenarios: probe-card degradation (extra random die fails), a radial yield gradient, and a map
   rotation (notch orientation change). The monitor must stay quiet on clean batches and block on
   drifted ones.
7. `model_card.py`: writes MODEL_CARD.md from results.json.

## Metrics (all produced by `python -m wafergate.run_all` into results/results.json)
1. **Classification**: macro-F1 and per-class recall on the lot-grouped test split, CNN vs gradient-boosting
   baseline (8 known classes). Rare-class recall called out separately.
2. **Unseen-pattern abstain**: AUROC and abstain recall/precision for held-out Donut wafers against
   in-distribution test wafers, at the validation-set threshold (target 5% in-distribution abstain).
   Baseline: max-softmax-probability thresholding.
3. **Governance**: gate decisions over a candidate set (baseline, CNN, and a deliberately degraded
   candidate trained without class weighting), and drift detection rate / false alarm rate over
   clean vs drifted batches, plus how far macro-F1 falls on the drifted batches the monitor blocked.

## Baselines
- Handcrafted features + gradient boosting (the classic Wu et al. 2015 approach, with a different learner).
- Max-softmax-probability abstain (Hendrycks and Gimpel 2017) for OOD.
- A no-monitor deployment for the drift scenario (what macro-F1 the fab would have gotten).
