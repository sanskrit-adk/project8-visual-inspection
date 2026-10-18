# Report notes - Project 8: Automated Visual Quality Inspection

_Generated 2026-10-18 by `scripts/10_make_report_notes.py`. Every number below is read from `results/metrics/`. Sections marked **FAST smoke run** come from 128px / 3-epoch / 25%-data runs and must be regenerated from the full run before submission._

**Environment:** Python 3.12.2, torch 2.14.1, device `cpu` (8 CPU threads), seed 42.

## 1. Data and exploration

- 7348 labelled images, all 300x300; raw defect share 57.3% (counts: {'test_OK': 262, 'test_Defect': 453, 'train_OK': 2875, 'train_Defect': 3758}).
- Near-duplicates (thumbnail correlation >= 0.976): 5163 images (70%) in 964 groups (largest 309); **279 groups span Kaggle's train and test folders** -> the published split leaks.
- 33 images have a near-copy with the other label (possible label noise).
- Shortcut baseline (logistic regression, group 5-fold CV): background-only ROC-AUC **0.839**, all simple stats 0.952. OK and defect photos differ in lighting.
- Figures: `results/figures/eda_samples.png`, `results/figures/eda_stat_distributions.png`, `results/figures/eda_duplicate_examples.png`, `results/figures/eda_cross_label_duplicates.png`, `results/figures/eda_shortcut_baseline.png`

## 2. Splits

| split | n | n_ok | n_defect | defect_frac | n_defect_groups |
|---|---|---|---|---|---|
| train | 2194 | 1975 | 219 | 10.0% | 219 |
| val | 471 | 424 | 47 | 10.0% | 47 |
| test | 471 | 424 | 47 | 10.0% | 47 |
| production | 349 | 314 | 35 | 10.0% | 35 |

- Split by duplicate group; leakage check: 0 shared groups, 0 shared files.
- Background-only baseline on the final test split: ROC-AUC **0.955**, AP 0.682 (prevalence 0.10).
- Sampler serves 49% defects per epoch. Figures: `results/figures/splits_composition.png`, `results/figures/augmentation_examples.png`

## 3. Models (validation) - FAST smoke run

| model | average_precision | roc_auc | recall_at_p95 | best_epoch | train_sec | params_trainable |
|---|---|---|---|---|---|---|
| scratch_cnn | 0.747 | 0.881 | 0.583 | 3 | 114 | 1,173,473 |
| resnet18_frozen | 0.569 | 0.782 | 0.333 | 3 | 43 | 513 |
| resnet18_finetune | 0.994 | 0.999 | 0.917 | 1 | 95 | 11,177,025 |
| mobilenet_v3_small | 0.879 | 0.985 | 0.417 | 3 | 67 | 1,518,881 |

- Background-only baseline on val: AP 0.615.

Data scarcity (val AP by number of training defects, FAST smoke run):

| n_defect | resnet18_finetune | scratch_cnn |
|---|---|---|
| 6 | 0.317 | 0.348 |
| 14 | 0.751 | 0.311 |
| 28 | 0.937 | 0.368 |
| 55 | 0.994 | 0.747 |

Figures: `results/figures/model_comparison_fast.png`, `results/figures/training_curves_fast.png`, `results/figures/data_scarcity_fast.png`

## 4. Test evaluation - FAST smoke run

Test: 118 images, 12 defects, 38 duplicate groups. Selected model (best val AP): **resnet18_finetune**. 95% CIs from a group bootstrap.

| model | AP [95% CI] | AUC [95% CI] | brier | test_precision_p95 | test_recall_p95 |
|---|---|---|---|---|---|
| scratch_cnn | 0.694 [0.429, 0.924] | 0.869 [0.691, 0.985] | 0.067 | 1.000 | 0.417 |
| resnet18_frozen | 0.632 [0.358, 0.870] | 0.818 [0.651, 0.949] | 0.265 | 1.000 | 0.417 |
| resnet18_finetune | 0.994 [0.966, 1.000] | 0.999 [0.995, 1.000] | 0.076 | 1.000 | 0.833 |
| mobilenet_v3_small | 0.941 [0.829, 1.000] | 0.991 [0.969, 1.000] | 0.048 | 1.000 | 0.500 |
| background_only | 0.542 [0.303, 0.878] | 0.939 [0.846, 0.986] | 0.077 | 1.000 | 0.083 |

Paired AP difference (selected minus other):

- vs scratch_cnn: +0.300 [+0.071, +0.567] (CI excludes 0)
- vs resnet18_frozen: +0.362 [+0.120, +0.641] (CI excludes 0)
- vs mobilenet_v3_small: +0.052 [-0.009, +0.167]
- vs background_only: +0.452 [+0.122, +0.685] (CI excludes 0)

At the val 95%-precision threshold (0.744): TP 10, FP 0, FN 2, TN 106.
Figures: `results/figures/pr_roc_curves_fast.png`, `results/figures/reliability_fast.png`, `results/figures/confusion_matrix_fast.png`, `results/figures/error_gallery_fast.png`

### 4b. Defect types (NEU-DET) - FAST smoke run

ResNet18 fine-tuned, 6 classes: accuracy **0.933**, macro-F1 0.931 on 90 test images. Most confused: inclusion <-> pitted_surface (3 images).

| type | precision | recall | f1-score |
|---|---|---|---|
| crazing | 1.000 | 0.933 | 0.966 |
| inclusion | 1.000 | 0.667 | 0.800 |
| patches | 1.000 | 1.000 | 1.000 |
| pitted_surface | 0.789 | 1.000 | 0.882 |
| rolled-in_scale | 1.000 | 1.000 | 1.000 |
| scratches | 0.882 | 1.000 | 0.938 |

Figure: `results/figures/neu_confusion_matrix_fast.png`

## 5. Cost-based threshold - FAST smoke run

Costs: shipped defect 50, false reject 1, line prevalence 10%. Threshold **0.443** for resnet18_finetune (chosen on val (smoothed cost curve)).

- Test: cost **59.4** per 1000 parts [11.1, 163.6], recall 1.000, false-reject rate 6.6%. Ship-everything costs 5000; reject-everything 900.
- Production split (used once): cost 57.7 per 1000, recall 1.000, false-reject rate 6.4% (87 parts, 9 defects).

| candidate | threshold | test_cost_per_1000 | test_recall | test_false_reject_rate |
|---|---|---|---|---|
| cost-optimal, smoothed (val) | 0.443 | 59.434 | 1.000 | 0.066 |
| cost-optimal, empirical (val) | 0.683 | 833.333 | 0.833 | 0.000 |
| 0.5 default | 0.500 | 16.981 | 1.000 | 0.019 |
| 95% precision (val) | 0.744 | 833.333 | 0.833 | 0.000 |
| Bayes + prior correction | 0.149 | 679.245 | 1.000 | 0.755 |

Figures: `results/figures/cost_curve_fast.png`, `results/figures/cost_sensitivity_fast.png`, `results/figures/model_costs_fast.png`

## 6. Latency - FAST smoke run

Device `cpu`, 8 threads, 128px; preprocessing 7.0 ms/image; line budget 500 ms/part.

| model | batch_size | median_ms_per_image | p95_ms_per_image | images_per_sec | headroom_vs_line | checkpoint_mb |
|---|---|---|---|---|---|---|
| scratch_cnn | 1 | 40.331 | 76.052 | 24.795 | 12x | 4.719 |
| scratch_cnn | 32 | 15.842 | 20.206 | 63.123 | 32x | 4.719 |
| resnet18_frozen | 1 | 44.694 | 109.111 | 22.374 | 11x | 44.788 |
| resnet18_frozen | 32 | 16.679 | 20.235 | 59.956 | 30x | 44.788 |
| resnet18_finetune | 1 | 59.397 | 214.141 | 16.836 | 8x | 44.789 |
| resnet18_finetune | 32 | 15.286 | 24.000 | 65.419 | 33x | 44.789 |
| mobilenet_v3_small | 1 | 38.235 | 88.525 | 26.154 | 13x | 6.213 |
| mobilenet_v3_small | 32 | 5.685 | 9.108 | 175.912 | 88x | 6.213 |

Figure: `results/figures/latency_fast.png`

## 7. Grad-CAM - FAST smoke run

- Castings (resnet18_finetune, images flagged as defect, n=19): 55% of heatmap mass on the part (part = 45% of the image; score **1.22**), 25% on the background ring (33% of the image; score 0.75). Score > 1 = preference for that region.
- Randomisation check: Spearman correlation between trained and random-weight maps = 0.160 (low = maps depend on what was learned).
- NEU: pointing game **58%** (peak inside an annotated box); 48% of mass inside boxes that cover 43% of the image.
- Figures: `results/figures/gradcam_castings_fast.png`, `results/figures/gradcam_part_vs_background_fast.png`, `results/figures/gradcam_neu_fast.png`

## 8. Robustness and shortcut tests - FAST smoke run

| model | original | part_only | background_only | background_equalized |
|---|---|---|---|---|
| mobilenet_v3_small | 0.941 | 0.427 | 0.387 | 0.689 |
| resnet18_finetune | 0.994 | 0.773 | 0.177 | 0.952 |
| resnet18_frozen | 0.632 | 0.482 | 0.184 | 0.641 |
| scratch_cnn | 0.694 | 0.485 | 0.270 | 0.543 |

Reading: a model using the part keeps its AP on *part only* and *background equalised*, and falls to chance on *background only*.

Deployed model (resnet18_finetune) at its fixed threshold 0.443 under imaging changes:

| family | level | average_precision | recall_at_thr | frr_at_thr | cost_per_1000_at_thr |
|---|---|---|---|---|---|
| original | 0 | 0.994 | 1.000 | 0.066 | 59 |
| brightness | 0.6 | 0.883 | 1.000 | 0.651 | 586 |
| brightness | 0.8 | 0.960 | 1.000 | 0.226 | 204 |
| brightness | 1.2 | 0.919 | 0.750 | 0.019 | 1267 |
| brightness | 1.4 | 0.760 | 0.500 | 0.000 | 2500 |
| brightness | 1.6 | 0.694 | 0.250 | 0.000 | 3750 |
| contrast | 0.5 | 0.988 | 1.000 | 0.566 | 509 |
| contrast | 0.7 | 0.988 | 1.000 | 0.179 | 161 |
| contrast | 1.5 | 0.952 | 0.917 | 0.038 | 451 |
| blur | 1 | 0.944 | 1.000 | 0.170 | 153 |
| blur | 2 | 0.866 | 1.000 | 0.377 | 340 |
| blur | 3 | 0.617 | 1.000 | 0.651 | 586 |
| rotation | 10 | 0.966 | 1.000 | 0.066 | 59 |
| rotation | 30 | 0.906 | 0.917 | 0.085 | 493 |
| rotation | 60 | 0.880 | 0.833 | 0.019 | 850 |
| part_only | 0 | 0.773 | 1.000 | 0.755 | 679 |
| background_only | 0 | 0.177 | 0.083 | 0.019 | 4600 |
| background_equalized | 0 | 0.952 | 1.000 | 0.302 | 272 |

Figures: `results/figures/robustness_shift_fast.png`, `results/figures/shortcut_tests_fast.png`, `results/figures/perturbation_examples_fast.png`

## 9. Demo

`streamlit run app/streamlit_app.py` - pick a held-out image or upload one; shows probability, accept/reject at the cost-based threshold, Grad-CAM, and a cost-ratio slider.

## Decision log

See `DECISIONS.md` (alternatives considered and evidence for each design choice).
