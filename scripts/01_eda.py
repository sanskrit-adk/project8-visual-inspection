"""Phase 1: download the casting data, index it, explore it, look for shortcuts.

Steps
  1. Download (kagglehub, cached under data/) and build data/index.csv.
  2. Per-image statistics: size, brightness, contrast, edge density, part vs background.
  3. Near-duplicate search on small thumbnails -> duplicate group ids
     (Phase 2 uses them for a group-aware split).
  4. Shortcut baseline: can a logistic regression on "dumb" statistics -
     especially background-only ones - separate defects from good parts?
  5. Figures + results/metrics/eda_summary.json + shortcut_baseline.json.

Run:  python scripts/01_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data import (build_index, download_casting, duplicate_groups,  # noqa: E402
                      image_stats, nearest_neighbours, radial_masks, resolve)
from src.utils import (CLASS_COLORS, CLASS_NAMES, INK_2, PALETTE, get_path,  # noqa: E402
                       load_config, plot_style, save_fig, save_json, set_seed)

# Feature sets for the shortcut baseline. Background-only features cannot see
# the casting, so any skill they show comes from how the photos were taken.
FEATURE_SETS = {
    "background_only": ["bg_mean", "bg_std"],
    "global_intensity": ["mean", "std"],
    "part_intensity": ["part_mean", "part_std"],
    "edge_density": ["edge_density"],
    "all_stats": ["mean", "std", "edge_density", "part_mean", "part_std", "bg_mean", "bg_std"],
}
STAT_COLS = ["mean", "std", "edge_density", "part_mean", "part_std", "bg_mean", "bg_std"]


# ---------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------
def compute_stats(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    rows, thumbs = [], []
    for p in tqdm(df["path"], desc="image stats"):
        s, t = image_stats(resolve(p), cfg)
        rows.append(s)
        thumbs.append(t)
    return pd.concat([df, pd.DataFrame(rows)], axis=1), np.stack(thumbs)


def duplicate_analysis(df: pd.DataFrame, thumbs: np.ndarray, cfg: dict) -> dict:
    """Pick a duplicate threshold from cross-class similarity, then group images."""
    y = df["label"].to_numpy()
    ok, de = np.where(y == 0)[0], np.where(y == 1)[0]

    # Same-class nearest neighbour (excluding self) vs cross-class nearest neighbour.
    within = np.empty(len(df), dtype=np.float32)
    within[ok], _ = nearest_neighbours(thumbs[ok], thumbs[ok], exclude_self=True)
    within[de], _ = nearest_neighbours(thumbs[de], thumbs[de], exclude_self=True)
    cross = np.empty(len(df), dtype=np.float32)
    cross[ok], _ = nearest_neighbours(thumbs[ok], thumbs[de], exclude_self=False)
    cross[de], _ = nearest_neighbours(thumbs[de], thumbs[ok], exclude_self=False)

    thr = cfg["data"]["dup_corr_threshold"]
    if thr == "auto":
        thr = float(np.percentile(cross, cfg["data"]["dup_auto_percentile"]))
    thr = float(thr)

    # Nearest neighbour over all images (any class): used to show real pairs and
    # to find copies that carry *different* labels (likely label noise).
    nn_corr, nn_idx = nearest_neighbours(thumbs, thumbs, exclude_self=True)

    groups = duplicate_groups(thumbs, thr)
    df["nn_corr_within"] = within
    df["nn_corr_cross"] = cross
    df["nn_index"] = nn_idx
    df["nn_corr"] = nn_corr
    df["dup_group"] = groups

    sizes = df.groupby("dup_group").size()
    multi = sizes[sizes > 1]
    in_multi = df["dup_group"].isin(multi.index)
    # Groups that span both of Kaggle's original splits = train/test leakage in the
    # published split.
    span = df[in_multi].groupby("dup_group")["orig_split"].nunique()
    mixed_label = df[in_multi].groupby("dup_group")["label"].nunique()
    # Direct pairs above threshold whose two images carry different labels.
    lab = df["label"].to_numpy()
    cross_pairs = (nn_corr >= thr) & (lab != lab[nn_idx])

    return {
        "threshold": thr,
        "threshold_source": "auto" if cfg["data"]["dup_corr_threshold"] == "auto" else "config",
        "within_class_nn_corr_median": float(np.median(within)),
        "cross_class_nn_corr_median": float(np.median(cross)),
        "n_groups": int(len(sizes)),
        "n_groups_size_gt1": int(len(multi)),
        "images_in_dup_groups": int(in_multi.sum()),
        "frac_images_in_dup_groups": float(in_multi.mean()),
        "largest_group": int(sizes.max()),
        "largest_group_frac": float(sizes.max() / len(df)),
        "groups_spanning_orig_train_and_test": int((span > 1).sum()),
        "images_in_groups_spanning_orig_split": int(df["dup_group"].isin(span[span > 1].index).sum()),
        "groups_with_both_labels": int((mixed_label > 1).sum()),
        "images_whose_nn_copy_has_other_label": int(cross_pairs.sum()),
        "note": ("Groups are connected components (single linkage), so chains of "
                 "augmented copies merge into large groups. That over-groups, which is "
                 "the safe direction for a leakage-free split."),
    }


def shortcut_baseline(df: pd.DataFrame, cfg: dict) -> dict:
    """5-fold CV logistic regression on each hand-crafted feature set.

    Folds keep duplicate groups together so copies of one part can't sit in both
    the fitting and scoring folds.
    """
    y = df["label"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    out = {"prevalence": float(y.mean()), "cv_folds": 5, "cv": "StratifiedGroupKFold(dup_group)",
           "note": "Chance level: ROC-AUC 0.5, AP = prevalence."}
    for name, cols in FEATURE_SETS.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        p = cross_val_predict(model, df[cols].to_numpy(), y, cv=cv, groups=df["dup_group"],
                              method="predict_proba")[:, 1]
        out[name] = {"features": cols, "roc_auc": float(roc_auc_score(y, p)),
                     "average_precision": float(average_precision_score(y, p))}
    return out


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def fig_class_balance(df, cfg):
    counts = df.groupby(["orig_split", "label"]).size().unstack("label").loc[["train", "test"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(counts))
    w = 0.38
    for i, lab in enumerate((0, 1)):
        bars = ax.bar(x + (i - 0.5) * w, counts[lab], w, color=CLASS_COLORS[lab],
                      label=CLASS_NAMES[lab])
        ax.bar_label(bars, padding=2, color=INK_2, fontsize=9)
    ax.set_xticks(x, ["Kaggle train", "Kaggle test"])
    ax.set_ylabel("Images")
    ax.set_title("Raw data has more defects than good parts")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save_fig(fig, "eda_class_balance", cfg)


def fig_samples(df, cfg, n=6):
    rng = np.random.default_rng(cfg["seed"])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for r, lab in enumerate((0, 1)):
        paths = df[df["label"] == lab]["path"].to_numpy()
        for c, p in enumerate(rng.choice(paths, n, replace=False)):
            ax = axes[r, c]
            ax.imshow(Image.open(resolve(p)).convert("L"), cmap="gray")
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True), s.set_color(CLASS_COLORS[lab]), s.set_linewidth(2)
        axes[r, 0].set_ylabel(CLASS_NAMES[lab], fontsize=11, color=CLASS_COLORS[lab])
    fig.suptitle("Random samples (top: OK, bottom: Defect)", fontweight="bold")
    save_fig(fig, "eda_samples", cfg)


def fig_masks(df, cfg):
    """Show which pixels count as 'part' and 'background' for the stats."""
    p = resolve(df[df["label"] == 1]["path"].iloc[0])
    g = np.asarray(Image.open(p).convert("L"))
    part, bg = radial_masks(*g.shape, cfg["data"]["part_radius_frac"],
                            cfg["data"]["background_radius_frac"])
    overlay = np.stack([g] * 3, -1).astype(np.float32) / 255
    overlay[part] = 0.65 * overlay[part] + 0.35 * np.array([0.16, 0.47, 0.84])
    overlay[bg] = 0.65 * overlay[bg] + 0.35 * np.array([0.92, 0.41, 0.20])
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(g, cmap="gray"), axes[0].set_title("Image")
    axes[1].imshow(overlay), axes[1].set_title("Part (blue) / background (orange)")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    save_fig(fig, "eda_part_background_masks", cfg)


def fig_stat_distributions(df, cfg):
    cols = [("mean", "Mean brightness"), ("std", "Global contrast (std)"),
            ("edge_density", "Edge density"), ("bg_mean", "Background brightness"),
            ("bg_std", "Background std"), ("part_mean", "Part brightness")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (c, title) in zip(axes.ravel(), cols):
        bins = np.histogram_bin_edges(df[c], bins=50)
        for lab in (0, 1):
            ax.hist(df.loc[df["label"] == lab, c], bins=bins, density=True, alpha=0.55,
                    color=CLASS_COLORS[lab], label=CLASS_NAMES[lab])
        ax.set_title(title)
        ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle("Per-image statistics by class: OK photos have a brighter background",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "eda_stat_distributions", cfg)


def fig_dup_hist(df, dup, cfg):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(min(df["nn_corr_cross"].min(), df["nn_corr_within"].min()), 1, 80)
    ax.hist(df["nn_corr_within"], bins=bins, alpha=0.6, color=PALETTE[2],
            label="Nearest neighbour, same class")
    ax.hist(df["nn_corr_cross"], bins=bins, alpha=0.6, color=PALETTE[6],
            label="Nearest neighbour, other class")
    ax.axvline(dup["threshold"], color=INK_2, ls="--", lw=1.2)
    ax.annotate(f"duplicate threshold {dup['threshold']:.3f}", (dup["threshold"], 0.95),
                xycoords=("data", "axes fraction"), xytext=(-6, 0),
                textcoords="offset points", ha="right", color=INK_2, fontsize=9)
    ax.set_xlabel("Thumbnail correlation with nearest neighbour")
    ax.set_ylabel("Images")
    ax.set_title("Same-class neighbours above the cross-class tail are likely copies")
    ax.legend(loc="upper left")
    save_fig(fig, "eda_duplicate_similarity", cfg)


def _pair_grid(df, idx, title, name, cfg):
    """Each column: an image (top) and its nearest neighbour (bottom)."""
    if len(idx) == 0:
        return
    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 4.8), squeeze=False)
    for c, i in enumerate(idx):
        for r, k in enumerate((i, df.at[i, "nn_index"])):
            row = df.loc[k]
            ax = axes[r, c]
            ax.imshow(Image.open(resolve(row["path"])).convert("L"), cmap="gray")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            ax.set_title(f"{CLASS_NAMES[row['label']]} · {row['orig_split']}", fontsize=8,
                         color=CLASS_COLORS[row["label"]], fontweight="normal")
        axes[1, c].set_xlabel(f"r = {df.at[i, 'nn_corr']:.4f}", fontsize=8)
    fig.suptitle(title, fontweight="bold")
    save_fig(fig, name, cfg)


def fig_dup_examples(df, dup, cfg, n=6):
    """Real nearest-neighbour pairs spread across the above-threshold range."""
    above = df[(df["nn_corr"] >= dup["threshold"]) & (df["label"] == df["label"].to_numpy()[
        df["nn_index"]])].sort_values("nn_corr")
    if len(above) == 0:
        return
    pick = above.index[np.linspace(0, len(above) - 1, n).astype(int)]
    _pair_grid(df, list(pick), "Near-duplicate pairs, from just above threshold to near-identical",
               "eda_duplicate_examples", cfg)


def fig_cross_label_pairs(df, dup, cfg, n=6):
    """Copies whose labels disagree - candidates for label noise."""
    lab = df["label"].to_numpy()
    m = (df["nn_corr"] >= dup["threshold"]) & (lab != lab[df["nn_index"]])
    # Keep one row per unordered pair.
    seen, pick = set(), []
    for i in df[m].sort_values("nn_corr", ascending=False).index:
        key = frozenset((i, df.at[i, "nn_index"]))
        if key not in seen:
            seen.add(key)
            pick.append(i)
        if len(pick) == n:
            break
    _pair_grid(df, pick, "Near-duplicates with different labels (possible label noise)",
               "eda_cross_label_duplicates", cfg)


def fig_shortcut(res, cfg):
    names = list(FEATURE_SETS)
    auc = [res[n]["roc_auc"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [PALETTE[1] if n == "background_only" else PALETTE[0] for n in names]
    bars = ax.barh(names, auc, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, color=INK_2, fontsize=9)
    ax.axvline(0.5, color=INK_2, ls="--", lw=1)
    ax.set_xlim(0.4, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("ROC-AUC (5-fold group CV, logistic regression)")
    ax.set_title("Shortcut baseline: how far do simple statistics get?")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "eda_shortcut_baseline", cfg)


# ---------------------------------------------------------------------
def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    plot_style()
    data_dir = get_path(cfg, "data_dir")
    metrics_dir = get_path(cfg, "metrics_dir")

    casting_dir = download_casting(cfg)
    print(f"Casting data: {casting_dir}")
    df = build_index(casting_dir, cfg)
    print(f"Indexed {len(df)} images")

    df, thumbs = compute_stats(df, cfg)
    dup = duplicate_analysis(df, thumbs, cfg)
    shortcut = shortcut_baseline(df, cfg)

    df.to_csv(data_dir / "index.csv", index=False)
    np.save(data_dir / "thumbs.npy", thumbs)

    fig_class_balance(df, cfg)
    fig_samples(df, cfg)
    fig_masks(df, cfg)
    fig_stat_distributions(df, cfg)
    fig_dup_hist(df, dup, cfg)
    fig_dup_examples(df, dup, cfg)
    fig_cross_label_pairs(df, dup, cfg)
    fig_shortcut(shortcut, cfg)

    by_class = df.groupby("label")[STAT_COLS].agg(["mean", "std"])
    summary = {
        "n_images": len(df),
        "counts": {f"{s}_{CLASS_NAMES[l]}": int(n)
                   for (s, l), n in df.groupby(["orig_split", "label"]).size().items()},
        "defect_fraction_raw": float(df["label"].mean()),
        "image_sizes": df.groupby(["width", "height"]).size().rename("n").reset_index()
                         .to_dict("records"),
        "image_modes": df["mode"].value_counts().to_dict(),
        "stats_by_class": {CLASS_NAMES[l]: {c: {"mean": float(by_class.loc[l, (c, "mean")]),
                                                "std": float(by_class.loc[l, (c, "std")])}
                                            for c in STAT_COLS} for l in (0, 1)},
        "duplicates": dup,
    }
    save_json(summary, metrics_dir / "eda_summary.json")
    save_json(shortcut, metrics_dir / "shortcut_baseline.json")

    print(f"\nClass counts: {summary['counts']}")
    print(f"Image sizes : {summary['image_sizes']}  modes: {summary['image_modes']}")
    print(f"Duplicates  : threshold {dup['threshold']:.4f} ({dup['threshold_source']}), "
          f"{dup['images_in_dup_groups']} images in {dup['n_groups_size_gt1']} groups "
          f"(largest {dup['largest_group']}), "
          f"{dup['groups_spanning_orig_train_and_test']} span Kaggle train/test, "
          f"{dup['groups_with_both_labels']} mix labels, "
          f"{dup['images_whose_nn_copy_has_other_label']} images' nearest copy has the other label")
    print("Shortcut baseline (ROC-AUC / AP):")
    for n in FEATURE_SETS:
        print(f"  {n:<18}{shortcut[n]['roc_auc']:.3f} / {shortcut[n]['average_precision']:.3f}")
    print(f"\nWrote {data_dir / 'index.csv'} and figures to results/figures/")


if __name__ == "__main__":
    main()
