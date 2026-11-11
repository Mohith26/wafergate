# Model card: wafer-defect-classifier (current champion)

**Champion:** `v3_cnn_weighted`  
**Data:** WM-811K (Wu, Jang, Chen 2015), MIR Lab public archive; 38717 train / 8283 val / 7964 test maps, split by lot (10738 lots, 0 lots shared across splits).  
**Split hash:** `b7a58412e63c43ef...`

## Intended use
Route electrical-wafer-sort maps to the right failure-pattern owner. Outputs one of 8 known patterns or ABSTAIN.
ABSTAIN means the map looks unlike anything in training and should go to an engineer.

## Not intended for
Final yield disposition without a human, other fabs or products without re-validation, or die-level root cause.

## Performance (lot-grouped test split, 8 known classes)

| metric | value |
|---|---|
| macro-F1 | 0.898 |
| accuracy | 0.946 |
| expected calibration error | 0.005 (temperature 0.83) |
| unseen-pattern (Donut) AUROC | 0.975 |
| Donut wafers abstained | 441/555 (79.5%) |
| in-distribution abstain rate | 4.3% |

| class | recall | test count |
|---|---|---|
| Center | 0.945 | 696 |
| Edge-Loc | 0.860 | 731 |
| Edge-Ring | 0.984 | 1206 |
| Loc | 0.812 | 575 |
| Near-full | 1.000 | 14 |
| Random | 0.918 | 98 |
| Scratch | 0.849 | 186 |
| none | 0.972 | 4458 |

## Release gate
Every candidate is judged against `policy.json`. Decisions this run:

- v1 `v1_gbm_handcrafted`: blocked: ood_auroc_min: 0.7193 vs floor 0.75
- v2 `v2_cnn_rushed_retrain`: blocked: per_class_recall_min: worst class Scratch recall 0.0968 vs floor 0.5
- v3 `v3_cnn_weighted`: promoted

## Monitoring
Batch drift monitor (PSI limit 0.25, abstain limit 0.100).

| scenario | batches blocked | macro-F1 if served anyway |
|---|---|---|
| clean | 0/20 | 0.899 |
| probe_card_wear | 20/20 | 0.853 |
| edge_bead_issue | 20/20 | 0.736 |
| tester_half_dropout | 20/20 | 0.744 |
| rotation_control | 0/20 | 0.895 |

## Known limits
- Near-full has only 14 test maps, so its recall is noisy.
- The `none` class was downsampled to keep training on a laptop CPU, so class priors differ from a real fab line.
- Drift scenarios are simulated perturbations of real maps, not recordings of a real tool excursion.
- Donut is the only held-out pattern tested; other unseen patterns may be harder or easier to catch.
