"""Phase 9b: build the static GitHub Pages site in docs/.

GitHub Pages serves static files only, so the deployed model is exported to ONNX
and run in the visitor's browser with ONNX Runtime Web. This script writes
everything docs/index.html needs:

  docs/model.onnx           deployed model, outputs P(defect); checked against PyTorch
  docs/data.json            threshold per cost ratio, metrics, figure list, samples
  docs/samples/             held-out test/production images + Grad-CAM overlays
  docs/figures/             result figures

Uses full-run results when they exist, otherwise the --fast ones (the page says which).

Run:  python scripts/09_export_site.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import build_transforms, resolve  # noqa: E402
from src.evaluation import choose_threshold, smoothed_cost_curve  # noqa: E402
from src.explain import compute_cams, make_cam, overlay  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.utils import ROOT, get_path, load_config, set_seed  # noqa: E402

DOCS = ROOT / "docs"
RATIOS = [1, 2, 5, 10, 20, 50, 100, 200]
# (figure name, section, caption)
FIGURES = [
    ("eda_samples", "Data", "Random OK (top) and defective (bottom) castings."),
    ("eda_stat_distributions", "Data", "Per-image statistics by class: OK photos have a brighter background."),
    ("eda_duplicate_examples", "Data", "Near-duplicate pairs: the dataset holds many re-shot copies of the same part."),
    ("eda_shortcut_baseline", "Data", "A model on simple image statistics - even background-only - separates the classes: a shortcut risk."),
    ("splits_composition", "Data", "Leakage-free splits by duplicate group, defects subsampled to 10%."),
    ("model_comparison", "Models", "Validation AP and recall at 95% precision vs the background-only baseline."),
    ("training_curves", "Models", "Training and validation curves."),
    ("data_scarcity", "Models", "How many defect examples each model needs."),
    ("pr_roc_curves", "Evaluation", "Test precision-recall and ROC curves."),
    ("reliability", "Evaluation", "Calibration: probabilities run high because training saw ~50% defects."),
    ("error_gallery", "Evaluation", "Hardest test images for the selected model."),
    ("neu_confusion_matrix", "Evaluation", "Per-defect-type performance on NEU-DET (6 steel defect types)."),
    ("cost_curve", "Threshold", "Expected cost per 1000 parts vs threshold; the smoothed validation curve picks the threshold."),
    ("cost_sensitivity", "Threshold", "How the threshold and outcomes move with the cost ratio."),
    ("model_costs", "Threshold", "Every model in cost terms at its own cost-optimal threshold."),
    ("latency", "Deployment", "Latency and throughput vs the line speed."),
    ("gradcam_castings", "Explainability", "Grad-CAM evidence for 'defect' on test castings."),
    ("gradcam_part_vs_background", "Explainability", "Share of Grad-CAM mass on the part vs the background ring."),
    ("gradcam_neu", "Explainability", "NEU Grad-CAM vs annotated defect boxes."),
    ("shortcut_tests", "Robustness", "Shortcut tests: grey out the background, the part, or equalise background brightness."),
    ("robustness_shift", "Robustness", "AP under brightness, contrast, blur and rotation changes."),
]


class ProbModel(torch.nn.Module):
    """Model + sigmoid, so the browser gets P(defect) directly."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return torch.sigmoid(self.m(x))


def pick_suffix(metrics: Path) -> str:
    return "" if (metrics / "threshold.json").exists() else "_fast"


def read(metrics, name, suffix):
    p = metrics / f"{name}{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def export_onnx(model, size, out: Path, check_x: torch.Tensor) -> float:
    """Export to a single-file ONNX model and return max |onnx - torch| on check_x."""
    import onnxruntime as ort

    pm = ProbModel(model).eval()
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(pm, (dummy,), str(out), input_names=["input"], output_names=["prob"],
                      dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}},
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(None, {"input": check_x[i:i + 1].numpy()})[0]
                          for i in range(len(check_x))])
    with torch.no_grad():
        ref = pm(check_x).numpy()
    return float(np.abs(got - ref).max())


def choose_samples(cfg, model_name, thr, suffix, n_each=6):
    """Per split: n_each defects + n_each OK at random, plus any errors (up to 3)."""
    metrics = get_path(cfg, "metrics_dir")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    test_pred = pd.read_csv(metrics / f"predictions_test{suffix}.csv")
    for split in ("test", "production"):
        df = pd.read_csv(get_path(cfg, "splits_dir") / f"{split}.csv")
        picked = []
        if split == "test":
            err = test_pred[(test_pred[model_name] >= thr) != test_pred["label"].astype(bool)]
            picked += err["path"].head(3).tolist()
        for lab in (1, 0):
            pool = [p for p in df[df["label"] == lab]["path"] if p not in picked]
            picked += list(rng.choice(pool, min(n_each, len(pool)), replace=False))
        for p in picked:
            r = df[df["path"] == p].iloc[0]
            rows.append({"path": p, "split": split, "label": int(r["label"]),
                         "filename": r["filename"]})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    metrics = get_path(cfg, "metrics_dir")
    suffix = pick_suffix(metrics)
    if not (metrics / f"threshold{suffix}.json").exists():
        sys.exit("No threshold file - run the pipeline first (scripts/run_all.py [--fast]).")
    th = read(metrics, "threshold", suffix)
    cfg["image"]["size"] = size = th["image_size"]
    model_name = th["model"]
    model, _ = load_checkpoint(get_path(cfg, "models_dir") / f"{model_name}{suffix}.pt", "cpu")
    print(f"Exporting {model_name} ({'FAST' if suffix else 'full'} run, {size}px)")

    if DOCS.exists():
        for sub in ("samples", "figures"):
            shutil.rmtree(DOCS / sub, ignore_errors=True)
    for sub in ("samples", "figures"):
        (DOCS / sub).mkdir(parents=True, exist_ok=True)
    (DOCS / ".nojekyll").touch()

    # Samples: copy image, torch probability, Grad-CAM overlay.
    samples = choose_samples(cfg, model_name, th["threshold"], suffix)
    tf = build_transforms(cfg, train=False)
    imgs = [Image.open(resolve(p)).convert("RGB") for p in samples["path"]]
    x = torch.stack([tf(im) for im in imgs])
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).numpy()
    cams = compute_cams(make_cam(model), x)
    out_samples = []
    for i, (im, r) in enumerate(zip(imgs, samples.itertuples())):
        stem = f"{r.split}_{Path(r.filename).stem}"
        im.save(DOCS / "samples" / f"{stem}.jpg", quality=90)
        gray = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32) / 255
        Image.fromarray(overlay(gray, cams[i])).resize((300, 300), Image.BILINEAR).save(
            DOCS / "samples" / f"{stem}_cam.png")
        out_samples.append({"image": f"samples/{stem}.jpg", "cam": f"samples/{stem}_cam.png",
                            "split": r.split, "label": "Defect" if r.label else "OK",
                            "filename": r.filename, "prob_torch": float(probs[i])})

    max_diff = export_onnx(model, size, DOCS / "model.onnx", x)
    print(f"ONNX vs PyTorch on {len(x)} samples: max |diff| = {max_diff:.2e}")
    assert max_diff < 1e-3, "ONNX export does not match PyTorch"

    # Threshold for each cost ratio on the slider (same method as Phase 5).
    val = pd.read_csv(metrics / f"predictions_val{suffix}.csv")
    grid = np.linspace(0, 1, cfg["evaluation"]["threshold_steps"])
    c_fp = cfg["evaluation"]["cost_false_reject"]
    thresholds = {str(r): choose_threshold(smoothed_cost_curve(
        val["label"].to_numpy(), val[model_name].to_numpy(), grid, r * c_fp, c_fp,
        cfg["data"]["defect_ratio"])) for r in RATIOS}

    # Figures.
    figs = []
    for name, section, caption in FIGURES:
        for s in (suffix, "", "_fast"):
            src = get_path(cfg, "figures_dir") / f"{name}{s}.png"
            if src.exists():
                shutil.copy(src, DOCS / "figures" / f"{name}.png")
                figs.append({"src": f"figures/{name}.png", "section": section, "caption": caption})
                break

    tm = read(metrics, "test_metrics", suffix)
    data = {
        "is_fast": bool(suffix), "model": model_name, "image_size": size,
        "mean": cfg["image"]["mean"], "std": cfg["image"]["std"],
        "threshold": th["threshold"], "cost_escaped_defect": th["cost_escaped_defect"],
        "cost_false_reject": th["cost_false_reject"], "line_prevalence": th["line_prevalence"],
        "thresholds_by_ratio": thresholds, "test": th["test"], "production": th["production"],
        "reference_cost_per_1000": th["reference_cost_per_1000"],
        "test_metrics": tm["metrics"] if tm else None,
        "n_test": tm["n_test"] if tm else None, "n_test_defects": tm["n_test_defects"] if tm else None,
        "eda": read(metrics, "eda_summary", ""), "shortcut_baseline": read(metrics, "shortcut_baseline", ""),
        "splits": read(metrics, "splits_summary", ""),
        "latency": read(metrics, "latency", suffix), "gradcam": read(metrics, "gradcam", suffix),
        "robustness_shortcut_ap": (read(metrics, "robustness", suffix) or {}).get("shortcut_ap"),
        "neu": {k: v for k, v in (read(metrics, "neu_per_type", suffix) or {}).items()
                if k in ("accuracy", "macro_f1", "classes", "most_confused_pair", "n_test")},
        "onnx_max_abs_diff": max_diff, "figures": figs, "samples": out_samples,
    }
    (DOCS / "data.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    mb = (DOCS / "model.onnx").stat().st_size / 1e6
    print(f"Wrote docs/: model.onnx ({mb:.1f} MB), {len(out_samples)} samples, {len(figs)} figures")


if __name__ == "__main__":
    main()
