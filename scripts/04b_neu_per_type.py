"""Phase 4b: per-defect-type performance on NEU-DET (6 steel surface defect types).

The casting data only says "defect / OK", so it cannot tell us which KINDS of
defect a CNN finds hard. NEU-DET labels six types, so we fine-tune the same
architecture (ResNet18) as a 6-class classifier and read the confusion matrix.

Splits: NEU's own validation folder is the test set; 15% of its train folder
(stratified) is held out for early stopping.

Outputs (suffix _fast in --fast mode):
  data/neu_index.csv, models/neu_resnet18.pt (reused by Grad-CAM in Phase 7),
  results/metrics/neu_per_type.json, results/figures/neu_confusion_matrix.png,
  results/figures/neu_samples.png

Run:  python scripts/04b_neu_per_type.py [--fast]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import (CastingDataset, build_neu_index, build_transforms,  # noqa: E402
                      download_neu, resolve)
from src.models import build_model, save_checkpoint  # noqa: E402
from src.train import fit_multiclass, predict_multiclass  # noqa: E402
from src.utils import (INK, apply_fast, get_device, get_path, load_config,  # noqa: E402
                       plot_style, save_fig, save_json, set_seed, tagged)

MODEL = "resnet18_finetune"


def make_neu_splits(index: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    seed = cfg["seed"]
    tr_all = index[index["orig_split"] == "train"]
    test = index[index["orig_split"] == "validation"]
    train, val = train_test_split(tr_all, test_size=0.15, stratify=tr_all["label"], random_state=seed)
    splits = {"train": train, "val": val, "test": test}
    if cfg["is_fast"]:
        f = cfg["fast"]["subset_fraction"]
        splits = {k: v.groupby("label").sample(frac=f, random_state=seed) for k, v in splits.items()}
    return {k: v.reset_index(drop=True) for k, v in splits.items()}


def make_neu_loaders(splits: dict, cfg: dict) -> dict:
    t = cfg["train"]
    g = torch.Generator().manual_seed(cfg["seed"])
    out = {}
    for name, df in splits.items():
        ds = CastingDataset(df, build_transforms(cfg, train=name == "train"), label_dtype=np.int64)
        out[name] = DataLoader(ds, batch_size=t["batch_size"], shuffle=name == "train",
                               generator=g if name == "train" else None,
                               num_workers=t["num_workers"],
                               persistent_workers=t["num_workers"] > 0)
    return out


def fig_samples(index, classes, cfg, n=4):
    rng = np.random.default_rng(cfg["seed"])
    fig, axes = plt.subplots(n, len(classes), figsize=(1.9 * len(classes), 1.9 * n + 0.4))
    for c, cls in enumerate(classes):
        paths = index[index["class"] == cls]["path"].to_numpy()
        for r, p in enumerate(rng.choice(paths, n, replace=False)):
            ax = axes[r, c]
            ax.imshow(Image.open(resolve(p)).convert("L"), cmap="gray")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        axes[0, c].set_title(cls, fontsize=9)
    fig.suptitle("NEU-DET: six steel surface defect types", fontweight="bold")
    save_fig(fig, "neu_samples", cfg)


def fig_confusion(cm, classes, acc, f1, cfg):
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("True type")
    ax.grid(False)
    ax.set_title(f"NEU defect types (test): accuracy {acc:.3f}, macro-F1 {f1:.3f}")
    save_fig(fig, tagged(cfg, "neu_confusion_matrix"), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = apply_fast(load_config(), args.fast)
    set_seed(cfg["seed"])
    plot_style()
    device = get_device()

    neu_dir = download_neu(cfg)
    index, classes = build_neu_index(neu_dir)
    index.to_csv(get_path(cfg, "data_dir") / "neu_index.csv", index=False)
    print(f"NEU: {len(index)} images, classes {classes}")
    fig_samples(index, classes, cfg)

    splits = make_neu_splits(index, cfg)
    print("  " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    loaders = make_neu_loaders(splits, cfg)

    set_seed(cfg["seed"])
    model = build_model(MODEL, num_outputs=len(classes))
    res = fit_multiclass(model, MODEL, loaders, cfg, device)
    save_checkpoint(model, MODEL, get_path(cfg, "models_dir") / f"{tagged(cfg, 'neu_resnet18')}.pt",
                    num_outputs=len(classes), classes=classes, image_size=cfg["image"]["size"],
                    best_epoch=res["best_epoch"], is_fast=cfg["is_fast"])

    p, y = predict_multiclass(model, loaders["test"], device)
    pred = p.argmax(1)
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    report = classification_report(y, pred, labels=range(len(classes)), target_names=classes,
                                   output_dict=True, zero_division=0)
    # Most-confused pair (off-diagonal, both directions summed).
    off = cm + cm.T
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)

    save_json({"model": MODEL, "classes": classes, "n_test": int(len(y)),
               "accuracy": acc, "macro_f1": f1,
               "per_class": {c: report[c] for c in classes},
               "confusion_matrix": cm, "most_confused_pair": [classes[i], classes[j], int(off[i, j])],
               "best_epoch": res["best_epoch"], "train_sec": res["train_sec"],
               "history": res["history"]},
              get_path(cfg, "metrics_dir") / f"{tagged(cfg, 'neu_per_type')}.json")
    fig_confusion(cm, classes, acc, f1, cfg)

    print(f"\nTest accuracy {acc:.3f}, macro-F1 {f1:.3f}")
    for c in classes:
        r = report[c]
        print(f"  {c:<17} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1-score']:.3f}")
    print(f"Most confused: {classes[i]} <-> {classes[j]} ({off[i, j]} images)")


if __name__ == "__main__":
    main()
