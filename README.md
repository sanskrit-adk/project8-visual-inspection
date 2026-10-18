# Project 8 — Automated Visual Quality Inspection

**Live site:** https://sanskrit-adk.github.io/project8-visual-inspection/ — in-browser demo + results.

A CNN system that flags defective cast parts from camera images, localises the defect with
Grad-CAM, chooses a decision threshold from asymmetric costs, and tests whether its accuracy
comes from real defect-finding skill or from quirks in how the data was collected.

## Setup

Requires Python 3.10+ (tested with 3.12).

```powershell
# Windows (PowerShell)
cd project8-visual-inspection
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/00_check_setup.py
```

```bash
# macOS / Linux
cd project8-visual-inspection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/00_check_setup.py
```

For a CUDA GPU, install the matching PyTorch build first (see https://pytorch.org/get-started/locally/),
then `pip install -r requirements.txt`. On Colab/Kaggle, torch is already installed.

**Kaggle access:** datasets are downloaded with `kagglehub`. If a download asks for credentials,
create an API token at kaggle.com → Settings → API and place `kaggle.json` in `~/.kaggle/`
(Windows: `C:\Users\<you>\.kaggle\kaggle.json`).

## Running each phase

Every training script accepts `--fast` (128px images, 3 epochs, 25% data subset; outputs get a `_fast` suffix) so it finishes
on a laptop CPU in minutes. Without `--fast`, settings are meant for a GPU.

| Phase | Command | Output |
|---|---|---|
| 0 Setup | `python scripts/00_check_setup.py` | device + versions, `results/metrics/environment.json` |
| 1 Data + EDA | `python scripts/01_eda.py` | `data/index.csv`, EDA figures, shortcut baseline |
| 2 Splits + loaders | `python scripts/02_splits.py` | `splits/*.csv`, augmentation figure |
| 3 Models | `python scripts/03_train_models.py [--fast]` | checkpoints, curves, `model_comparison.csv` |
| 4 Evaluation | `python scripts/04_evaluate.py [--fast]` | PR/ROC, confusion, error gallery |
| 4b NEU per-type | `python scripts/04b_neu_per_type.py [--fast]` | 6-class confusion matrix |
| 5 Threshold | `python scripts/05_threshold.py [--fast]` | cost curve, `threshold.json` |
| 6 Latency | `python scripts/06_latency.py [--fast]` | latency table + chart |
| 7 Grad-CAM | `python scripts/07_gradcam.py [--fast]` | heatmap grids, sanity score |
| 8 Robustness | `python scripts/08_robustness.py [--fast]` | shift results, shortcut tests |
| 9 Demo | `streamlit run app/streamlit_app.py` | web demo |
| 9b Website | `python scripts/09_export_site.py` | `docs/` (GitHub Pages: ONNX model runs in the browser) |
| 10 Report notes | `python scripts/10_make_report_notes.py` | `report/REPORT_NOTES.md` |

Run everything in order with `python scripts/run_all.py [--fast]` (stops at the first failing
phase; resume with `--from <phase>`, e.g. `--from 4` after training on a GPU). On a laptop CPU the
fast run takes ~15–20 min; the full run takes several hours on CPU and minutes on a GPU.

Fast-mode outputs carry a `_fast` suffix. The demo and `10_make_report_notes.py` use full-run
results when they exist and fall back to the fast ones, labelling which they used.

## Layout

```
config.yaml        all paths + hyperparameters (seed 42)
src/               reusable code (data, models, training, evaluation, ...)
scripts/           one entry script per phase
splits/            train/val/test/production CSVs shared by the team (production = untouched final check)
results/figures/   every figure (PNG, dpi 150)
results/metrics/   every number (JSON/CSV) — the report only quotes these
DECISIONS.md       decision log
report/            auto-generated report notes + contributions
```

> Note: this folder lives under OneDrive. `data/`, `models/` and `venv/` hold many files;
> if syncing is slow, pause OneDrive or exclude those folders.
