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

## 3. Models (validation) - full run

| model | average_precision | roc_auc | recall_at_p95 | best_epoch | train_sec | params_trainable |
|---|---|---|---|---|---|---|
| scratch_cnn | 1.000 | 1.000 | 1.000 | 8 | 3851 | 1,173,473 |
| resnet18_frozen | 0.980 | 0.996 | 0.936 | 22 | 1594 | 513 |
| resnet18_finetune | 0.991 | 0.998 | 0.979 | 1 | 877 | 11,177,025 |
| mobilenet_v3_small | 1.000 | 1.000 | 1.000 | 4 | 376 | 1,518,881 |

- Background-only baseline on val: AP 0.558.

Data scarcity (val AP by number of training defects, FAST smoke run):

| n_defect | resnet18_finetune | scratch_cnn |
|---|---|---|
| 6 | 0.317 | 0.348 |
| 14 | 0.751 | 0.311 |
| 28 | 0.937 | 0.368 |
| 55 | 0.994 | 0.747 |

Figures: `results/figures/model_comparison.png`, `results/figures/training_curves.png`, `results/figures/data_scarcity_fast.png`

## 4. Test evaluation - full run

Test: 471 images, 47 defects, 94 duplicate groups. Selected model (best val AP): **mobilenet_v3_small**. 95% CIs from a group bootstrap.

| model | AP [95% CI] | AUC [95% CI] | brier | test_precision_p95 | test_recall_p95 |
|---|---|---|---|---|---|
| scratch_cnn | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.017 | 0.959 | 1.000 |
| resnet18_frozen | 0.948 [0.882, 0.997] | 0.984 [0.956, 1.000] | 0.077 | 1.000 | 0.809 |
| resnet18_finetune | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 | 1.000 | 1.000 |
| mobilenet_v3_small | 0.999 [0.994, 1.000] | 1.000 [0.999, 1.000] | 0.007 | 0.758 | 1.000 |
| background_only | 0.682 [0.525, 0.860] | 0.955 [0.904, 0.983] | 0.058 | 1.000 | 0.085 |

Paired AP difference (selected minus other):

- vs scratch_cnn: -0.001 [-0.006, +0.000]
- vs resnet18_frozen: +0.051 [+0.001, +0.116] (CI excludes 0)
- vs resnet18_finetune: -0.001 [-0.006, +0.000]
- vs background_only: +0.316 [+0.140, +0.473] (CI excludes 0)

At the val 95%-precision threshold (0.000): TP 47, FP 15, FN 0, TN 409.
Figures: `results/figures/pr_roc_curves.png`, `results/figures/reliability.png`, `results/figures/confusion_matrix.png`, `results/figures/error_gallery.png`

### 4b. Defect types (NEU-DET) - full run

ResNet18 fine-tuned, 6 classes: accuracy **1.000**, macro-F1 1.000 on 360 test images. Most confused: crazing <-> crazing (0 images).

| type | precision | recall | f1-score |
|---|---|---|---|
| crazing | 1.000 | 1.000 | 1.000 |
| inclusion | 1.000 | 1.000 | 1.000 |
| patches | 1.000 | 1.000 | 1.000 |
| pitted_surface | 1.000 | 1.000 | 1.000 |
| rolled-in_scale | 1.000 | 1.000 | 1.000 |
| scratches | 1.000 | 1.000 | 1.000 |

Figure: `results/figures/neu_confusion_matrix.png`

## 5. Cost-based threshold - full run

Costs: shipped defect 50, false reject 1, line prevalence 10%. Threshold **0.001** for mobilenet_v3_small (chosen on val (smoothed cost curve)).

- Test: cost **12.7** per 1000 parts [1.5, 38.9], recall 1.000, false-reject rate 1.4%. Ship-everything costs 5000; reject-everything 900.
- Production split (used once): cost 22.9 per 1000, recall 1.000, false-reject rate 2.5% (349 parts, 35 defects).

| candidate | threshold | test_cost_per_1000 | test_recall | test_false_reject_rate |
|---|---|---|---|---|
| cost-optimal, smoothed (val) | 0.001 | 12.736 | 1.000 | 0.014 |
| cost-optimal, empirical (val) | 0.004 | 2.123 | 1.000 | 0.002 |
| 0.5 default | 0.500 | 319.149 | 0.936 | 0.000 |
| 95% precision (val) | 0.000 | 31.840 | 1.000 | 0.035 |
| Bayes + prior correction | 0.149 | 319.149 | 0.936 | 0.000 |

Figures: `results/figures/cost_curve.png`, `results/figures/cost_sensitivity.png`, `results/figures/model_costs.png`

## 6. Latency - full run

Device `cpu`, 8 threads, 224px; preprocessing 3.8 ms/image; line budget 500 ms/part.

| model | batch_size | median_ms_per_image | p95_ms_per_image | images_per_sec | headroom_vs_line | checkpoint_mb |
|---|---|---|---|---|---|---|
| scratch_cnn | 1 | 23.301 | 31.740 | 42.916 | 21x | 4.718 |
| scratch_cnn | 32 | 27.695 | 34.151 | 36.107 | 18x | 4.718 |
| resnet18_frozen | 1 | 30.406 | 35.331 | 32.888 | 16x | 44.788 |
| resnet18_frozen | 32 | 16.475 | 19.855 | 60.699 | 30x | 44.788 |
| resnet18_finetune | 1 | 20.435 | 27.155 | 48.935 | 24x | 44.788 |
| resnet18_finetune | 32 | 16.172 | 20.708 | 61.837 | 31x | 44.788 |
| mobilenet_v3_small | 1 | 9.114 | 11.455 | 109.716 | 55x | 6.211 |
| mobilenet_v3_small | 32 | 3.277 | 3.699 | 305.182 | 153x | 6.211 |

Figure: `results/figures/latency.png`

## 7. Grad-CAM - full run

- Castings (mobilenet_v3_small, images flagged as defect, n=53): 57% of heatmap mass on the part (part = 45% of the image; score **1.26**), 17% on the background ring (34% of the image; score 0.51). Score > 1 = preference for that region.
- Randomisation check: Spearman correlation between trained and random-weight maps = 0.102 (low = maps depend on what was learned).
- NEU: pointing game **82%** (peak inside an annotated box); 52% of mass inside boxes that cover 43% of the image.
- Figures: `results/figures/gradcam_castings.png`, `results/figures/gradcam_part_vs_background.png`, `results/figures/gradcam_neu.png`

## 8. Robustness and shortcut tests - full run

| model | original | part_only | background_only | background_equalized |
|---|---|---|---|---|
| mobilenet_v3_small | 0.999 | 0.816 | 0.357 | 0.999 |
| resnet18_finetune | 1.000 | 0.945 | 0.356 | 1.000 |
| resnet18_frozen | 0.948 | 0.839 | 0.163 | 0.842 |
| scratch_cnn | 1.000 | 0.948 | 0.235 | 0.980 |

Reading: a model using the part keeps its AP on *part only* and *background equalised*, and falls to chance on *background only*.

Deployed model (mobilenet_v3_small) at its fixed threshold 0.001 under imaging changes:

| family | level | average_precision | recall_at_thr | frr_at_thr | cost_per_1000_at_thr |
|---|---|---|---|---|---|
| original | 0 | 0.999 | 1.000 | 0.014 | 13 |
| brightness | 0.6 | 1.000 | 1.000 | 0.038 | 34 |
| brightness | 0.8 | 1.000 | 1.000 | 0.009 | 8 |
| brightness | 1.2 | 0.999 | 1.000 | 0.014 | 13 |
| brightness | 1.4 | 0.988 | 0.979 | 0.021 | 125 |
| brightness | 1.6 | 0.930 | 0.915 | 0.064 | 483 |
| contrast | 0.5 | 0.999 | 1.000 | 0.033 | 30 |
| contrast | 0.7 | 1.000 | 1.000 | 0.017 | 15 |
| contrast | 1.5 | 1.000 | 1.000 | 0.733 | 660 |
| blur | 1 | 1.000 | 1.000 | 0.002 | 2 |
| blur | 2 | 1.000 | 1.000 | 0.028 | 25 |
| blur | 3 | 0.973 | 1.000 | 0.847 | 762 |
| rotation | 10 | 1.000 | 0.979 | 0.000 | 106 |
| rotation | 30 | 0.999 | 0.979 | 0.000 | 106 |
| rotation | 60 | 0.997 | 0.979 | 0.005 | 111 |
| part_only | 0 | 0.816 | 1.000 | 0.762 | 686 |
| background_only | 0 | 0.357 | 1.000 | 1.000 | 900 |
| background_equalized | 0 | 0.999 | 1.000 | 0.028 | 25 |

Figures: `results/figures/robustness_shift.png`, `results/figures/shortcut_tests.png`, `results/figures/perturbation_examples.png`

## 9. Demo

`streamlit run app/streamlit_app.py` - pick a held-out image or upload one; shows probability, accept/reject at the cost-based threshold, Grad-CAM, and a cost-ratio slider.

## Decision log

See `DECISIONS.md` (alternatives considered and evidence for each design choice).
