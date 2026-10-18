"""Phase 3: train and compare the four models, plus a data-scarcity experiment.

Steps
  1. Train scratch_cnn, resnet18_frozen, resnet18_finetune, mobilenet_v3_small
     with early stopping on validation AP; save the best checkpoint of each.
  2. Compare them on the validation split (AP, ROC-AUC, recall at fixed precision,
     size, training time) against the background-only shortcut baseline.
     The test split is NOT used here - it is reserved for Phase 4.
  3. Data scarcity: retrain scratch_cnn and resnet18_finetune on 10/25/50% of the
     training set to see how much pretraining helps when defects are scarce.

Outputs (suffix _fast in --fast mode):
  models/<model>.pt, results/metrics/model_comparison.csv, training_history.json,
  data_scarcity.csv, results/figures/training_curves.png, model_comparison.png,
  data_scarcity.png

Run:  python scripts/03_train_models.py [--fast] [--models a b ...] [--skip-scarcity]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data import fast_subset, load_split, make_loaders  # noqa: E402
from src.models import MODEL_NAMES, build_model, count_params  # noqa: E402
from src.train import fit  # noqa: E402
from src.utils import (INK_2, PALETTE, apply_fast, get_device, get_path,  # noqa: E402
                       load_config, plot_style, save_fig, save_json, set_seed, tagged)

MODEL_COLORS = {m: PALETTE[i] for i, m in enumerate(MODEL_NAMES)}
SCARCITY_MODELS = ["scratch_cnn", "resnet18_finetune"]


def shortcut_reference(cfg: dict) -> dict:
    """Background-only logistic regression, fit on train, scored on val."""
    tr, va = load_split(cfg, "train"), load_split(cfg, "val")
    cols = ["bg_mean", "bg_std"]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[cols], tr["label"])
    p = m.predict_proba(va[cols])[:, 1]
    return {"average_precision": float(average_precision_score(va["label"], p)),
            "roc_auc": float(roc_auc_score(va["label"], p))}


def run_models(names, cfg, device) -> tuple[pd.DataFrame, dict]:
    loaders = make_loaders(cfg, splits=("train", "val"))
    rows, histories = [], {}
    models_dir = get_path(cfg, "models_dir")
    for name in names:
        print(f"\n=== {name} ===")
        set_seed(cfg["seed"])  # same init/batch order for every model
        model = build_model(name)
        total, trainable = count_params(model)
        res = fit(model, name, loaders, cfg, device, ckpt_path=models_dir / f"{tagged(cfg, name)}.pt")
        histories[name] = res["history"]
        rows.append({"model": name, **res["val"], "val_loss": res["val_loss"],
                     "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                     "train_sec": res["train_sec"], "params_total": total,
                     "params_trainable": trainable})
    return pd.DataFrame(rows), histories


def run_scarcity(cfg, device, full_rows: pd.DataFrame) -> pd.DataFrame:
    """Retrain on stratified subsets of train; the 100% point reuses the main run."""
    full_train = load_split(cfg, "train")
    rows = []
    for frac in cfg["train"]["scarcity_fractions"]:
        for name in SCARCITY_MODELS:
            if frac >= 1.0 and name in set(full_rows["model"]):
                r = full_rows.set_index("model").loc[name]
                rows.append({"model": name, "fraction": 1.0, "n_train": len(full_train),
                             "n_defect": int(full_train["label"].sum()),
                             "average_precision": r["average_precision"], "roc_auc": r["roc_auc"]})
                continue
            sub = fast_subset(full_train, frac, cfg["seed"]).reset_index(drop=True)
            print(f"\n=== scarcity {frac:.0%}: {name} ({len(sub)} images, "
                  f"{int(sub['label'].sum())} defects) ===")
            set_seed(cfg["seed"])
            loaders = make_loaders(cfg, splits=("train", "val"), train_df=sub)
            res = fit(build_model(name), name, loaders, cfg, device, verbose=False)
            print(f"  val AP {res['val']['average_precision']:.4f}  "
                  f"AUC {res['val']['roc_auc']:.4f}  (best epoch {res['best_epoch']})")
            rows.append({"model": name, "fraction": frac, "n_train": len(sub),
                         "n_defect": int(sub["label"].sum()),
                         "average_precision": res["val"]["average_precision"],
                         "roc_auc": res["val"]["roc_auc"]})
    return pd.DataFrame(rows).sort_values(["model", "fraction"])


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def fig_curves(histories, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for name, h in histories.items():
        h = pd.DataFrame(h)
        c = MODEL_COLORS[name]
        axes[0].plot(h["epoch"], h["train_loss"], color=c, label=name)
        axes[1].plot(h["epoch"], h["val_loss"], color=c)
        axes[2].plot(h["epoch"], h["val_ap"], color=c, marker="o", ms=3)
    for ax, t in zip(axes, ["Train loss (balanced batches)", "Validation loss", "Validation AP"]):
        ax.set_title(t)
        ax.set_xlabel("Epoch")
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, tagged(cfg, "training_curves"), cfg)


def fig_comparison(df, ref, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    y = np.arange(len(df))
    colors = [MODEL_COLORS[m] for m in df["model"]]
    for ax, col, title in [(axes[0], "average_precision", "Validation AP"),
                           (axes[1], "recall_at_p95", "Recall at 95% precision (val)")]:
        bars = ax.barh(y, df[col], color=colors)
        ax.bar_label(bars, fmt="%.3f", padding=3, color=INK_2, fontsize=9)
        ax.set_yticks(y, df["model"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].axvline(ref["average_precision"], color=INK_2, ls="--", lw=1,
                    label=f"background-only baseline ({ref['average_precision']:.3f})")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=8.5)
    axes[1].set_yticks(y, [""] * len(y))
    fig.tight_layout()
    save_fig(fig, tagged(cfg, "model_comparison"), cfg)


def fig_scarcity(df, cfg):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name, g in df.groupby("model"):
        ax.plot(g["n_defect"], g["average_precision"], marker="o", color=MODEL_COLORS[name],
                label=name)
        last = g.iloc[-1]
        ax.annotate(name, (last["n_defect"], last["average_precision"]), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=MODEL_COLORS[name])
    ax.set_xscale("log")
    ticks = sorted(df["n_defect"].unique())
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Defect images in training set (log scale)")
    ax.set_ylabel("Validation AP")
    ax.set_title("How many defect examples does each model need?")
    ax.margins(x=0.25)
    save_fig(fig, tagged(cfg, "data_scarcity"), cfg)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="128px, 3 epochs, 25%% data (CPU smoke run)")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--skip-scarcity", action="store_true")
    args = ap.parse_args()

    cfg = apply_fast(load_config(), args.fast)
    set_seed(cfg["seed"])
    plot_style()
    device = get_device()
    metrics_dir = get_path(cfg, "metrics_dir")
    print(f"Device {device} | image {cfg['image']['size']}px | epochs {cfg['train']['epochs']} "
          f"| imbalance {cfg['train']['imbalance']} | fast={cfg['is_fast']}")

    df, histories = run_models(args.models, cfg, device)
    ref = shortcut_reference(cfg)
    df.to_csv(metrics_dir / f"{tagged(cfg, 'model_comparison')}.csv", index=False)
    save_json({"histories": histories, "background_only_val": ref},
              metrics_dir / f"{tagged(cfg, 'training_history')}.json")
    fig_curves(histories, cfg)
    fig_comparison(df, ref, cfg)

    print("\nValidation comparison:")
    cols = ["model", "average_precision", "roc_auc", "recall_at_p90", "recall_at_p95",
            "best_epoch", "train_sec", "params_trainable"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"background-only baseline: AP {ref['average_precision']:.3f}, "
          f"AUC {ref['roc_auc']:.3f}")

    if not args.skip_scarcity:
        sc = run_scarcity(cfg, device, df)
        sc.to_csv(metrics_dir / f"{tagged(cfg, 'data_scarcity')}.csv", index=False)
        fig_scarcity(sc, cfg)
        print("\nData scarcity (val AP):")
        print(sc.pivot(index="fraction", columns="model", values="average_precision")
                .to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
