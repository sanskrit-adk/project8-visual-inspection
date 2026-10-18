"""Phase 2: leakage-free splits, defect subsampling, and data loaders.

Steps
  1. Deal whole duplicate groups (from Phase 1) into train/val/test/production.
  2. In each split keep every OK image and subsample defects to `defect_ratio`.
  3. Check for leakage: no duplicate group and no file in two splits.
  4. Re-run the background-only shortcut baseline on the final splits
     (fit on train, score on test) - the number the CNN must clearly beat.
  5. Smoke-test the loaders and save an augmentation figure.

Outputs: splits/{train,val,test,production}.csv, results/metrics/splits_summary.json,
         results/figures/splits_composition.png, results/figures/augmentation_examples.png

Run:  python scripts/02_splits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data import (SPLITS, assign_splits, build_transforms, make_loaders,  # noqa: E402
                      pos_weight, resolve, split_fractions, subsample_defects)
from src.utils import (CLASS_COLORS, CLASS_NAMES, INK_2, apply_fast, get_path,  # noqa: E402
                       load_config, plot_style, save_fig, save_json, set_seed)

KEEP_COLS = ["path", "filename", "class", "label", "orig_split", "dup_group", "label_conflict",
             "mean", "std", "edge_density", "part_mean", "part_std", "bg_mean", "bg_std"]


def make_splits(index: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    lab = index["label"].to_numpy()
    # The image's nearest copy (above the duplicate threshold) carries the other label.
    thr = index.attrs.get("dup_threshold")
    index["label_conflict"] = (index["nn_corr"] >= thr) & (lab != lab[index["nn_index"]])
    index["split"] = assign_splits(index, cfg)
    out = {}
    for i, name in enumerate(SPLITS):
        part = index[index["split"] == name]
        out[name] = subsample_defects(part, cfg["data"]["defect_ratio"], cfg["seed"] + i)[KEEP_COLS]
    return out


def leakage_checks(splits: dict) -> dict:
    groups = {n: set(df["dup_group"]) for n, df in splits.items()}
    paths = {n: set(df["path"]) for n, df in splits.items()}
    shared_groups, shared_paths = 0, 0
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            shared_groups += len(groups[a] & groups[b])
            shared_paths += len(paths[a] & paths[b])
    assert shared_groups == 0, f"{shared_groups} duplicate groups appear in two splits"
    assert shared_paths == 0, f"{shared_paths} files appear in two splits"
    return {"duplicate_groups_shared_between_splits": shared_groups,
            "files_shared_between_splits": shared_paths}


def shortcut_on_splits(splits: dict) -> dict:
    """Background-only and all-stats logistic regression: train -> test."""
    sets = {"background_only": ["bg_mean", "bg_std"],
            "all_stats": ["mean", "std", "edge_density", "part_mean", "part_std",
                          "bg_mean", "bg_std"]}
    tr, te = splits["train"], splits["test"]
    out = {"test_prevalence": float(te["label"].mean())}
    for name, cols in sets.items():
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        m.fit(tr[cols], tr["label"])
        p = m.predict_proba(te[cols])[:, 1]
        out[name] = {"roc_auc": float(roc_auc_score(te["label"], p)),
                     "average_precision": float(average_precision_score(te["label"], p))}
    return out


def sampler_balance(cfg: dict) -> dict:
    """Defect fraction the training loader actually serves over one epoch."""
    loader = make_loaders(cfg, splits=("train",))["train"]
    labels = loader.dataset.labels
    if loader.sampler is not None and hasattr(loader.sampler, "weights"):
        idx = np.fromiter(iter(loader.sampler), dtype=int)
        served = float(labels[idx].mean())
    else:
        served = float(labels.mean())
    return {"imbalance": cfg["train"]["imbalance"], "train_defect_frac": float(labels.mean()),
            "defect_frac_served_per_epoch": served}


def smoke_test(cfg: dict) -> dict:
    """One batch from each loader in fast mode: shapes, dtypes, value range."""
    fast_cfg = apply_fast(load_config(), True)
    loaders = make_loaders(fast_cfg)
    out = {}
    for name, ld in loaders.items():
        x, y = next(iter(ld))
        out[name] = {"n_images_fast": len(ld.dataset), "batch_shape": list(x.shape),
                     "label_dtype": str(y.dtype), "x_min": float(x.min()), "x_max": float(x.max())}
    return out


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def fig_composition(splits, cfg):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(SPLITS))
    ok = np.array([(splits[n]["label"] == 0).sum() for n in SPLITS])
    de = np.array([(splits[n]["label"] == 1).sum() for n in SPLITS])
    ax.bar(x, ok, color=CLASS_COLORS[0], label=CLASS_NAMES[0])
    ax.bar(x, de, bottom=ok, color=CLASS_COLORS[1], label=CLASS_NAMES[1])
    for xi, o, d in zip(x, ok, de):
        ax.annotate(f"{o} OK · {d} defect", (xi, o + d), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=INK_2, fontsize=8.5)
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("Images")
    ax.set_title(f"Splits by duplicate group, defects subsampled to "
                 f"{cfg['data']['defect_ratio']:.0%}")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, (ok + de).max() * 1.12)
    save_fig(fig, "splits_composition", cfg)


def _denorm(x, cfg):
    m = torch.tensor(cfg["image"]["mean"]).view(3, 1, 1)
    s = torch.tensor(cfg["image"]["std"]).view(3, 1, 1)
    return (x * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def fig_augmentation(splits, cfg, n_aug=6):
    torch.manual_seed(cfg["seed"])
    tr = splits["train"]
    eval_t, train_t = build_transforms(cfg, train=False), build_transforms(cfg, train=True)
    fig, axes = plt.subplots(2, n_aug + 1, figsize=(1.9 * (n_aug + 1), 4.3))
    for r, lab in enumerate((0, 1)):
        img = Image.open(resolve(tr[tr["label"] == lab]["path"].iloc[0])).convert("RGB")
        views = [eval_t(img)] + [train_t(img) for _ in range(n_aug)]
        for c, v in enumerate(views):
            ax = axes[r, c]
            ax.imshow(_denorm(v, cfg))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        axes[r, 0].set_title("original", fontsize=9, fontweight="normal")
        axes[r, 0].set_ylabel(CLASS_NAMES[lab], color=CLASS_COLORS[lab], fontsize=11)
    fig.suptitle("Training augmentation: flips, ±15° rotation, brightness/contrast, blur",
                 fontweight="bold")
    save_fig(fig, "augmentation_examples", cfg)


# ---------------------------------------------------------------------
def main():
    cfg = load_config()
    cfg["is_fast"] = False
    set_seed(cfg["seed"])
    plot_style()

    index_path = get_path(cfg, "data_dir") / "index.csv"
    if not index_path.exists():
        sys.exit("data/index.csv not found - run scripts/01_eda.py first.")
    index = pd.read_csv(index_path)
    eda = pd.read_json(get_path(cfg, "metrics_dir") / "eda_summary.json", typ="series")
    index.attrs["dup_threshold"] = eda["duplicates"]["threshold"]

    splits = make_splits(index, cfg)
    leak = leakage_checks(splits)
    splits_dir = get_path(cfg, "splits_dir")
    for name, df in splits.items():
        df.to_csv(splits_dir / f"{name}.csv", index=False)

    used = pd.concat(splits.values())
    per_split = {}
    for name, df in splits.items():
        per_split[name] = {
            "n": len(df), "n_ok": int((df["label"] == 0).sum()),
            "n_defect": int(df["label"].sum()), "defect_frac": float(df["label"].mean()),
            "n_dup_groups": int(df["dup_group"].nunique()),
            "n_defect_groups": int(df.loc[df["label"] == 1, "dup_group"].nunique()),
            "from_kaggle_test": int((df["orig_split"] == "test").sum()),
            "label_conflicts": int(df["label_conflict"].sum()),
        }
    summary = {
        "defect_ratio_target": cfg["data"]["defect_ratio"],
        "split_fracs_target": split_fractions(cfg),
        "ok_frac_actual": {n: s["n_ok"] / sum(v["n_ok"] for v in per_split.values())
                           for n, s in per_split.items()},
        "splits": per_split,
        "images_used": len(used),
        "defects_discarded_by_subsampling": int(index["label"].sum() - used["label"].sum()),
        "train_pos_weight": pos_weight(splits["train"]),
        "leakage": leak,
        "shortcut_train_to_test": shortcut_on_splits(splits),
        "sampler": sampler_balance(cfg),
        "loader_smoke_test_fast": smoke_test(cfg),
    }
    save_json(summary, get_path(cfg, "metrics_dir") / "splits_summary.json")

    fig_composition(splits, cfg)
    fig_augmentation(splits, cfg)

    print(f"{'split':<11}{'n':>6}{'OK':>6}{'def':>6}{'def%':>7}{'groups':>8}{'def groups':>11}")
    for n, s in per_split.items():
        print(f"{n:<11}{s['n']:>6}{s['n_ok']:>6}{s['n_defect']:>6}{s['defect_frac']:>7.1%}"
              f"{s['n_dup_groups']:>8}{s['n_defect_groups']:>11}")
    print(f"Leakage     : {leak}")
    sc = summary["shortcut_train_to_test"]
    print(f"Shortcut    : background-only test AUC {sc['background_only']['roc_auc']:.3f} "
          f"/ AP {sc['background_only']['average_precision']:.3f} "
          f"(prevalence {sc['test_prevalence']:.3f}); all-stats AUC "
          f"{sc['all_stats']['roc_auc']:.3f} / AP {sc['all_stats']['average_precision']:.3f}")
    sb = summary["sampler"]
    print(f"Sampler     : train defect frac {sb['train_defect_frac']:.3f} -> served "
          f"{sb['defect_frac_served_per_epoch']:.3f} per epoch")
    for n, s in summary["loader_smoke_test_fast"].items():
        print(f"Smoke ({n:<5}): {s['n_images_fast']} imgs, batch {s['batch_shape']}, "
              f"x in [{s['x_min']:.2f}, {s['x_max']:.2f}]")


if __name__ == "__main__":
    main()
