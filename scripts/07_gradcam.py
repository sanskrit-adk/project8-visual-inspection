"""Phase 7: Grad-CAM - where does the model look?

Castings (selected model, test split):
  * Heatmap grids for caught defects, missed defects and false alarms.
  * Sanity score: share of Grad-CAM mass on the PART vs on the BACKGROUND ring,
    divided by each region's share of the image (1.0 = no preference). Phase 1
    found background brightness separates the classes; a model exploiting that
    would put its mass on the background.
  * Randomisation check (Adebayo et al., 2018): heatmaps from the trained model
    should differ from those of the same architecture with random weights. If
    they look alike, the map shows image edges, not what the model learned.
NEU-DET (Phase 4b model, test split):
  * Pointing game: does the heatmap peak fall inside an annotated defect box?
  * Mass inside boxes vs the boxes' share of the image.

Outputs (suffix _fast in --fast mode):
  results/metrics/gradcam.json, results/figures/gradcam_castings.png,
  gradcam_part_vs_background.png, gradcam_neu.png

Run:  python scripts/07_gradcam.py [--fast] [--max-images 200]
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
from scipy.stats import spearmanr  # noqa: E402

from src.data import build_transforms, radial_masks, resolve  # noqa: E402
from src.explain import compute_cams, energy_fraction, make_cam, overlay, peak_in  # noqa: E402
from src.models import build_model, load_checkpoint  # noqa: E402
from src.utils import (CLASS_COLORS, INK_2, PALETTE, apply_fast, get_device,  # noqa: E402
                       get_path, load_config, plot_style, save_fig, save_json, set_seed,
                       tagged)


def load_images(paths, cfg):
    """Eval-transformed tensors + the matching resized greyscale images for display."""
    tf = build_transforms(cfg, train=False)
    size = cfg["image"]["size"]
    xs, grays = [], []
    for p in paths:
        img = Image.open(resolve(p)).convert("RGB")
        xs.append(tf(img))
        grays.append(np.asarray(img.convert("L").resize((size, size)), dtype=np.float32) / 255)
    return torch.stack(xs), np.stack(grays)


# ---------------------------------------------------------------------
# Castings
# ---------------------------------------------------------------------
def castings(cfg, device, max_images, rng):
    metrics_dir = get_path(cfg, "metrics_dir")
    thr_info = json.loads((metrics_dir / f"{tagged(cfg, 'threshold')}.json").read_text())
    model_name, thr = thr_info["model"], thr_info["threshold"]
    preds = pd.read_csv(metrics_dir / f"{tagged(cfg, 'predictions_test')}.csv")
    model, _ = load_checkpoint(get_path(cfg, "models_dir") / f"{tagged(cfg, model_name)}.pt", device)

    # Keep every defect and error; fill up with OK parts to max_images.
    p = preds[model_name]
    preds["pred"] = (p >= thr).astype(int)
    keep = preds[(preds["label"] == 1) | (preds["pred"] == 1)]
    ok = preds.drop(keep.index)
    n_ok = max(0, min(len(ok), max_images - len(keep)))
    sel = pd.concat([keep, ok.sample(n_ok, random_state=cfg["seed"])]).reset_index(drop=True)

    x, grays = load_images(sel["path"], cfg)
    x = x.to(device)
    cams = compute_cams(make_cam(model), x)
    size = cfg["image"]["size"]
    part, bg = radial_masks(size, size, cfg["data"]["part_radius_frac"],
                            cfg["data"]["background_radius_frac"])
    sel["part_energy"] = energy_fraction(cams, part)
    sel["bg_energy"] = energy_fraction(cams, bg)
    sel["peak_on_part"] = peak_in(cams, part)

    # Randomisation check on a subset: same architecture, random weights.
    n_r = min(64, len(sel))
    torch.manual_seed(cfg["seed"])
    rand_model = build_model(model_name, pretrained=False).to(device).eval()
    rand_cams = compute_cams(make_cam(rand_model), x[:n_r])
    # A random-weight model often gives a constant (all-zero) map; correlation is
    # undefined there, so those are skipped and counted.
    flat = np.array([a.std() < 1e-8 or b.std() < 1e-8 for a, b in zip(cams[:n_r], rand_cams)])
    rho = [spearmanr(a.ravel(), b.ravel())[0]
           for a, b, f in zip(cams[:n_r], rand_cams, flat) if not f]

    def grp(mask):
        d = sel[mask]
        return {"n": int(len(d)), "part_energy": float(d["part_energy"].mean()) if len(d) else None,
                "bg_energy": float(d["bg_energy"].mean()) if len(d) else None,
                "peak_on_part": float(d["peak_on_part"].mean()) if len(d) else None}

    y, pr = sel["label"], sel["pred"]
    res = {
        "model": model_name, "threshold": thr, "n_images": int(len(sel)),
        "part_area_frac": float(part.mean()), "bg_area_frac": float(bg.mean()),
        "all": grp(sel.index == sel.index),
        "true_positive": grp((y == 1) & (pr == 1)), "false_negative": grp((y == 1) & (pr == 0)),
        "false_positive": grp((y == 0) & (pr == 1)), "true_negative": grp((y == 0) & (pr == 0)),
        "randomization_spearman_mean": float(np.nanmean(rho)),
        "randomization_n": n_r, "randomization_constant_maps": int(flat.sum()),
    }
    # Score on images the model flags as defects: there the map shows the evidence
    # it acted on. On parts it calls OK there is little defect evidence, and the
    # min-max-scaled map mostly shows noise, so they would dilute the score.
    res["flagged"] = grp(pr == 1)
    res["sanity_score_part"] = res["flagged"]["part_energy"] / res["part_area_frac"]
    res["sanity_score_bg"] = res["flagged"]["bg_energy"] / res["bg_area_frac"]
    fig_casting_grid(sel, cams, grays, model_name, thr, cfg, rng)
    fig_part_vs_bg(sel, res, cfg)
    return res


def fig_casting_grid(sel, cams, grays, model_name, thr, cfg, rng, n=5):
    groups = [("Caught defects", (sel["label"] == 1) & (sel["pred"] == 1), 1),
              ("Missed defects", (sel["label"] == 1) & (sel["pred"] == 0), 1),
              ("False alarms", (sel["label"] == 0) & (sel["pred"] == 1), 0),
              ("Correct OK", (sel["label"] == 0) & (sel["pred"] == 0), 0)]
    fig, axes = plt.subplots(len(groups), n, figsize=(2.1 * n, 2.3 * len(groups) + 0.5))
    for r, (title, mask, lab) in enumerate(groups):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = rng.choice(idx, n, replace=False)
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            if c >= len(idx):
                ax.axis("off")
                if c == 0:
                    ax.text(0.5, 0.5, "none", ha="center", va="center", color=INK_2,
                            transform=ax.transAxes)
                continue
            i = idx[c]
            ax.imshow(overlay(grays[i], cams[i]))
            ax.set_title(f"p={sel.at[i, model_name]:.2f}", fontsize=8, fontweight="normal",
                         color=INK_2)
            for s in ax.spines.values():
                s.set_visible(True), s.set_color(CLASS_COLORS[lab]), s.set_linewidth(2)
        axes[r, 0].set_ylabel(f"{title}\n(n={mask.sum()})", fontsize=9, color=CLASS_COLORS[lab])
    fig.suptitle(f"Grad-CAM for 'defect' - {model_name}, thr {thr:.3f} (bright = evidence)",
                 fontweight="bold")
    save_fig(fig, tagged(cfg, "gradcam_castings"), cfg)


def fig_part_vs_bg(sel, res, cfg):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    groups = ["true_positive", "false_negative", "false_positive", "true_negative"]
    labels = ["caught\ndefects", "missed\ndefects", "false\nalarms", "correct\nOK"]
    x = np.arange(len(groups))
    w = 0.38
    part = [res[g]["part_energy"] or 0 for g in groups]
    bg = [res[g]["bg_energy"] or 0 for g in groups]
    b1 = ax.bar(x - w / 2, part, w, color=PALETTE[2], label="on part")
    b2 = ax.bar(x + w / 2, bg, w, color=PALETTE[7], label="on background ring")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8, color=INK_2)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8, color=INK_2)
    ax.axhline(res["part_area_frac"], color=PALETTE[2], ls=":", lw=1)
    ax.axhline(res["bg_area_frac"], color=PALETTE[7], ls=":", lw=1)
    ax.set_xticks(x, [f"{l}\n(n={res[g]['n']})" for l, g in zip(labels, groups)])
    ax.set_ylabel("Share of Grad-CAM mass")
    ax.set_title("Where the evidence sits (dotted = share of image area)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    save_fig(fig, tagged(cfg, "gradcam_part_vs_background"), cfg)


# ---------------------------------------------------------------------
# NEU
# ---------------------------------------------------------------------
def neu(cfg, device, max_images):
    ckpt = get_path(cfg, "models_dir") / f"{tagged(cfg, 'neu_resnet18')}.pt"
    idx_path = get_path(cfg, "data_dir") / "neu_index.csv"
    if not ckpt.exists() or not idx_path.exists():
        print("  (skip NEU: run scripts/04b_neu_per_type.py first)")
        return None
    model, meta = load_checkpoint(ckpt, device)
    classes = meta["classes"]
    test = pd.read_csv(idx_path)
    test = test[test["orig_split"] == "validation"]
    per = max(1, max_images // len(classes))
    test = test.groupby("label").sample(min(per, test["label"].value_counts().min()),
                                        random_state=cfg["seed"]).reset_index(drop=True)
    size = cfg["image"]["size"]
    x, grays = load_images(test["path"], cfg)
    cams = compute_cams(make_cam(model), x.to(device), class_idx=test["label"].tolist())

    masks, area = [], []
    for b in test["boxes"]:
        m = np.zeros((size, size), bool)
        for x0, y0, x1, y1 in json.loads(b):   # boxes are in 200px coordinates
            s = size / 200
            m[int(y0 * s):int(np.ceil(y1 * s)), int(x0 * s):int(np.ceil(x1 * s))] = True
        masks.append(m)
        area.append(m.mean())
    masks, area = np.stack(masks), np.array(area)
    inside = np.array([c[m].sum() / (c.sum() + 1e-8) for c, m in zip(cams, masks)])
    hit = peak_in(cams, list(masks))
    test["box_area"], test["box_energy"], test["pointing_hit"] = area, inside, hit
    per_class = test.groupby("class").agg(n=("label", "size"), box_area=("box_area", "mean"),
                                          box_energy=("box_energy", "mean"),
                                          pointing=("pointing_hit", "mean"))
    fig_neu(test, cams, grays, masks, classes, cfg)
    return {"n_images": int(len(test)), "pointing_game_acc": float(hit.mean()),
            "box_energy_mean": float(inside.mean()), "box_area_mean": float(area.mean()),
            "energy_over_area": float(inside.mean() / area.mean()),
            "per_class": per_class.reset_index().to_dict("records")}


def fig_neu(test, cams, grays, masks, classes, cfg, n=3):
    fig, axes = plt.subplots(n, len(classes), figsize=(2 * len(classes), 2.1 * n + 0.5))
    for c, cls in enumerate(classes):
        idx = np.where(test["class"] == cls)[0][:n]
        for r in range(n):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            if r >= len(idx):
                ax.axis("off")
                continue
            i = idx[r]
            ax.imshow(overlay(grays[i], cams[i]))
            ax.contour(masks[i], levels=[0.5], colors=[PALETTE[2]], linewidths=1.2)
        axes[0, c].set_title(cls, fontsize=9)
    fig.suptitle("NEU Grad-CAM for the true class (green = annotated defect box)",
                 fontweight="bold")
    save_fig(fig, tagged(cfg, "gradcam_neu"), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-images", type=int, default=200)
    args = ap.parse_args()

    cfg = apply_fast(load_config(), args.fast)
    set_seed(cfg["seed"])
    plot_style()
    device = get_device()
    rng = np.random.default_rng(cfg["seed"])
    try:
        cast = castings(cfg, device, args.max_images, rng)
    except FileNotFoundError as e:
        sys.exit(f"{e.filename} not found - run scripts 04 and 05 "
                 f"{'with --fast ' if args.fast else ''}first.")
    neu_res = neu(cfg, device, args.max_images)
    save_json({"castings": cast, "neu": neu_res},
              get_path(cfg, "metrics_dir") / f"{tagged(cfg, 'gradcam')}.json")

    a = cast["flagged"]
    print(f"Castings ({cast['model']}, {cast['n_images']} images; scores on the {a['n']} flagged "
          f"as defect): part mass {a['part_energy']:.3f} (area {cast['part_area_frac']:.3f}, "
          f"score {cast['sanity_score_part']:.2f}), background mass {a['bg_energy']:.3f} "
          f"(area {cast['bg_area_frac']:.3f}, score {cast['sanity_score_bg']:.2f}), "
          f"peak on part {a['peak_on_part']:.0%}")
    for g in ("true_positive", "false_negative", "false_positive", "true_negative"):
        d = cast[g]
        if d["n"]:
            print(f"  {g:<15} n={d['n']:>3}  part {d['part_energy']:.3f}  bg {d['bg_energy']:.3f}")
    print(f"Randomisation check: Spearman(trained, random-weight CAM) = "
          f"{cast['randomization_spearman_mean']:.3f} (low = map depends on learned weights)")
    if neu_res:
        print(f"NEU ({neu_res['n_images']} images): pointing game {neu_res['pointing_game_acc']:.2%}, "
              f"box mass {neu_res['box_energy_mean']:.3f} vs box area {neu_res['box_area_mean']:.3f}")


if __name__ == "__main__":
    main()
