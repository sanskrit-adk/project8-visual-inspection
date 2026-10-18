"""Phase 8: robustness to imaging changes, and shortcut tests.

Distribution shift (test split): brightness, contrast, blur and rotation at the
levels in config.yaml -> AP for every model, plus the deployed model's recall,
false-reject rate and cost at its Phase 5 threshold (the threshold is NOT re-tuned:
a line wouldn't re-tune it either).

Shortcut tests - is the model reading the part or the photo setup?
  part_only            background replaced by one grey      -> AP should stay high
  background_only      whole casting replaced by one grey   -> AP should fall to chance
  background_equalized background brightness made identical -> AP should stay high
Phase 1 showed background brightness alone reaches ROC-AUC ~0.95 on test, so a
model that learned that shortcut would fail the last two.

Outputs (suffix _fast in --fast mode):
  results/metrics/robustness.csv, robustness.json
  results/figures/robustness_shift.png, shortcut_tests.png, perturbation_examples.png

Run:  python scripts/08_robustness.py [--fast]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import robustness as R  # noqa: E402
from src.data import CastingDataset, build_transforms, load_split  # noqa: E402
from src.evaluation import confusion_at, expected_cost  # noqa: E402
from src.models import MODEL_NAMES, load_checkpoint  # noqa: E402
from src.utils import (INK_2, PALETTE, apply_fast, get_device, get_path, load_config,  # noqa: E402
                       plot_style, save_fig, save_json, set_seed, tagged)

COLORS = {m: PALETTE[i] for i, m in enumerate(MODEL_NAMES)}
SHORT = {"scratch_cnn": "scratch", "resnet18_frozen": "r18frz", "resnet18_finetune": "r18ft",
         "mobilenet_v3_small": "mnv3"}
SHORTCUTS = ["original", "part_only", "background_only", "background_equalized"]


def conditions(cfg, train):
    """(family, level, perturb) for every test condition; level is the x-axis value."""
    r = cfg["robustness"]
    fill = cfg["augmentation"]["rotation_fill"]
    bg_fill = int(round(train["bg_mean"].median()))
    part_fill = int(round(train["part_mean"].median()))
    out = [("original", 0, lambda im: im)]
    out += [("brightness", f, R.brightness(f)) for f in r["brightness_factors"]]
    out += [("contrast", f, R.contrast(f)) for f in r["contrast_factors"]]
    out += [("blur", s, R.blur(s)) for s in r["blur_sigmas"]]
    out += [("rotation", d, R.rotation(d, fill)) for d in r["rotation_degs"]]
    out += [("part_only", 0, R.part_only(cfg, bg_fill)),
            ("background_only", 0, R.background_only(cfg, part_fill)),
            ("background_equalized", 0, R.background_equalized(cfg, bg_fill))]
    return out, {"background_fill": bg_fill, "part_fill": part_fill, "rotation_fill": fill}


@torch.no_grad()
def score_all(models, loader, device):
    probs = {m: [] for m in models}
    labels = []
    for x, y in loader:
        x = x.to(device)
        for m, net in models.items():
            probs[m].append(torch.sigmoid(net(x)).cpu().numpy())
        labels.append(y.numpy())
    return {m: np.concatenate(v) for m, v in probs.items()}, np.concatenate(labels).astype(int)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def fig_shift(df, cfg):
    fams = [("brightness", "Brightness factor", 1.0), ("contrast", "Contrast factor", 1.0),
            ("blur", "Gaussian blur sigma (px)", 0.0), ("rotation", "Rotation (degrees)", 0.0)]
    orig = df[df["family"] == "original"].set_index("model")["average_precision"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, (fam, xlabel, ident) in zip(axes, fams):
        for m, g in df[df["family"] == fam].groupby("model"):
            g = pd.concat([g[["level", "average_precision"]],
                           pd.DataFrame({"level": [ident], "average_precision": [orig[m]]})])
            g = g.sort_values("level")
            ax.plot(g["level"], g["average_precision"], marker="o", ms=4, color=COLORS[m], label=m)
        ax.axvline(ident, color=INK_2, lw=0.8, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_title(fam.capitalize())
    axes[0].set_ylabel("Test AP")
    prev = df["prevalence"].iloc[0]
    for ax in axes:
        ax.axhline(prev, color=INK_2, lw=0.8, ls="--")
    axes[0].annotate("chance", (axes[0].get_xlim()[0], prev), xytext=(2, 3),
                     textcoords="offset points", fontsize=8, color=INK_2)
    axes[-1].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Robustness to imaging changes (dotted line = unchanged images)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, tagged(cfg, "robustness_shift"), cfg)


def fig_shortcuts(df, cfg):
    d = df[df["family"].isin(SHORTCUTS)]
    piv = d.pivot(index="model", columns="family", values="average_precision")[SHORTCUTS]
    piv = piv.loc[[m for m in MODEL_NAMES if m in piv.index]]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    x = np.arange(len(SHORTCUTS))
    w = 0.8 / len(piv)
    for i, (m, row) in enumerate(piv.iterrows()):
        bars = ax.bar(x + (i - (len(piv) - 1) / 2) * w, row.values, w, color=COLORS[m], label=m)
        ax.bar_label(bars, fmt="%.2f", padding=1, fontsize=7, color=INK_2)
    prev = df["prevalence"].iloc[0]
    ax.axhline(prev, color=INK_2, lw=1, ls="--")
    ax.annotate(f"chance (AP = prevalence {prev:.2f})", (x[-1] + 0.45, prev), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    ax.set_xticks(x, ["original", "part only\n(background greyed)",
                      "background only\n(casting greyed)", "background\nbrightness equalised"])
    ax.set_ylabel("Test AP")
    ax.set_ylim(0, 1.1)
    ax.set_title("Shortcut tests: is the model reading the part or the photo setup?")
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    save_fig(fig, tagged(cfg, "shortcut_tests"), cfg)


def fig_examples(path, conds, cfg):
    pick = ["original", ("brightness", 0.6), ("brightness", 1.6), ("contrast", 0.5),
            ("blur", 3.0), ("rotation", 30), "part_only", "background_only", "background_equalized"]
    lookup = {(f, l): fn for f, l, fn in conds}
    lookup.update({f: fn for f, l, fn in conds if f in SHORTCUTS})
    img = Image.open(path).convert("RGB")
    items = [(k if isinstance(k, str) else f"{k[0]} {k[1]:g}", lookup.get(k)) for k in pick]
    items = [(t, fn) for t, fn in items if fn is not None]
    fig, axes = plt.subplots(1, len(items), figsize=(1.9 * len(items), 2.4))
    for ax, (t, fn) in zip(axes, items):
        ax.imshow(fn(img))
        ax.set_title(t.replace("_", " "), fontsize=8, fontweight="normal")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    fig.suptitle("Test conditions applied to one image", fontweight="bold")
    save_fig(fig, tagged(cfg, "perturbation_examples"), cfg)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = apply_fast(load_config(), args.fast)
    set_seed(cfg["seed"])
    plot_style()
    device = get_device()
    metrics_dir = get_path(cfg, "metrics_dir")
    ev = cfg["evaluation"]

    thr_path = metrics_dir / f"{tagged(cfg, 'threshold')}.json"
    if not thr_path.exists():
        sys.exit(f"{thr_path.name} not found - run scripts/05_threshold.py"
                 f"{' --fast' if args.fast else ''} first.")
    thr_info = json.loads(thr_path.read_text())
    deployed, thr = thr_info["model"], thr_info["threshold"]

    models = {}
    for name in MODEL_NAMES:
        p = get_path(cfg, "models_dir") / f"{tagged(cfg, name)}.pt"
        if p.exists():
            models[name] = load_checkpoint(p, device)[0]
    test = load_split(cfg, "test")
    base = CastingDataset(test, build_transforms(cfg, train=False))
    conds, fills = conditions(cfg, load_split(cfg, "train"))

    rows = []
    for fam, level, fn in conds:
        loader = DataLoader(R.PerturbedDataset(base, fn), batch_size=cfg["train"]["batch_size"],
                            num_workers=cfg["train"]["num_workers"])
        probs, y = score_all(models, loader, device)
        for m, p in probs.items():
            row = {"family": fam, "level": level, "model": m, "prevalence": float(y.mean()),
                   "average_precision": average_precision_score(y, p),
                   "roc_auc": roc_auc_score(y, p)}
            if m == deployed:
                c = confusion_at(y, p, thr)
                row.update({"recall_at_thr": c["recall"], "frr_at_thr": c["false_reject_rate"],
                            "cost_per_1000_at_thr": 1000 * expected_cost(
                                y, p, thr, ev["cost_escaped_defect"], ev["cost_false_reject"],
                                cfg["data"]["defect_ratio"])})
            rows.append(row)
        d = next(r for r in rows[::-1] if r["model"] == deployed) if deployed in probs else None
        msg = f"  {fam:<21}{level:>5g}  " + "  ".join(
            f"{SHORT[m]} {average_precision_score(y, p):.3f}" for m, p in probs.items())
        if d:
            msg += f"  | {deployed} @thr: recall {d['recall_at_thr']:.2f} FRR {d['frr_at_thr']:.2f}"
        print(msg)

    df = pd.DataFrame(rows)
    df.to_csv(metrics_dir / f"{tagged(cfg, 'robustness')}.csv", index=False)
    orig = df[df["family"] == "original"].set_index("model")["average_precision"]
    worst = (df[~df["family"].isin(SHORTCUTS)].groupby("model")["average_precision"].min())
    sc = df[df["family"].isin(SHORTCUTS)].pivot(index="model", columns="family",
                                                values="average_precision")
    save_json({"deployed_model": deployed, "threshold": thr, "fills": fills,
               "original_ap": orig.to_dict(), "worst_shift_ap": worst.to_dict(),
               "shortcut_ap": sc.to_dict("index"), "rows": df.to_dict("records")},
              metrics_dir / f"{tagged(cfg, 'robustness')}.json")
    fig_shift(df, cfg)
    fig_shortcuts(df, cfg)
    defect = test[test["label"] == 1]["path"].iloc[0]
    fig_examples(base.paths[test.index[test["path"] == defect][0]], conds, cfg)

    print("\nShortcut tests (test AP):")
    print(sc[SHORTCUTS].to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
