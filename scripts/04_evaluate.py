"""Phase 4: evaluate every trained model on the held-out test split.

Steps
  1. Predict val + test with each checkpoint from Phase 3 (and the background-only
     shortcut baseline); save per-image predictions for Phases 5 and 8.
  2. Test metrics with 95% group-bootstrap intervals: AP, ROC-AUC, Brier score.
  3. Operating points: thresholds that reach 90% / 95% precision are chosen on
     VALIDATION and then applied unchanged to test - the honest way to report
     what precision/recall the line would actually get.
  4. Paired bootstrap: is the selected model's AP really higher than each other's?
  5. Figures: PR + ROC curves, reliability diagram, confusion matrix, error gallery.

The selected model is the one with the best VALIDATION AP (Phase 3), never the
best test score.

Outputs (suffix _fast in --fast mode):
  results/metrics/test_metrics.csv, test_metrics.json, predictions_val.csv, predictions_test.csv
  results/figures/pr_roc_curves.png, reliability.png, confusion_matrix.png, error_gallery.png

Run:  python scripts/04_evaluate.py [--fast] [--n-boot 2000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss,  # noqa: E402
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data import load_split, make_loaders, resolve  # noqa: E402
from src.evaluation import (ci, confusion_at, group_bootstrap, predict,  # noqa: E402
                            recall_at_precision)
from src.models import MODEL_NAMES, load_checkpoint  # noqa: E402
from src.utils import (CLASS_COLORS, INK, INK_2, PALETTE, apply_fast, get_device,  # noqa: E402
                       get_path, load_config, plot_style, save_fig, save_json, set_seed,
                       tagged)

BASELINE = "background_only"
COLORS = {**{m: PALETTE[i] for i, m in enumerate(MODEL_NAMES)}, BASELINE: "#8a8984"}
ID_COLS = ["path", "filename", "label", "dup_group", "label_conflict", "orig_split"]


# ---------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------
def collect_predictions(cfg, device) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    loaders = make_loaders(cfg, splits=("val", "test"))
    val = loaders["val"].dataset.df[ID_COLS].copy()
    test = loaders["test"].dataset.df[ID_COLS].copy()
    models_dir = get_path(cfg, "models_dir")
    found = []
    for name in MODEL_NAMES:
        path = models_dir / f"{tagged(cfg, name)}.pt"
        if not path.exists():
            print(f"  (skip {name}: {path.name} not found)")
            continue
        model, meta = load_checkpoint(path, device)
        if meta.get("image_size") != cfg["image"]["size"]:
            sys.exit(f"{path.name} was trained at {meta.get('image_size')}px but this run uses "
                     f"{cfg['image']['size']}px - use the matching --fast setting.")
        for df, split in ((val, "val"), (test, "test")):
            p, y, _ = predict(model, loaders[split], device)
            assert (y == df["label"].to_numpy()).all(), "loader order changed"
            df[name] = p
        found.append(name)
        print(f"  predicted {name}")

    # Shortcut baseline: background statistics only, fit on train.
    cols = ["bg_mean", "bg_std"]
    tr = load_split(cfg, "train")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[cols], tr["label"])
    for df, split in ((val, "val"), (test, "test")):
        feats = load_split(cfg, split).set_index("path").loc[df["path"], cols]
        df[BASELINE] = lr.predict_proba(feats)[:, 1]
    return val, test, found


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def test_metrics(val, test, names, selected, cfg, n_boot) -> tuple[pd.DataFrame, dict]:
    y, g = test["label"].to_numpy(), test["dup_group"].to_numpy()
    scores = {n: test[n].to_numpy() for n in names}
    boot_ap = group_bootstrap(y, scores, g, average_precision_score, n_boot, cfg["seed"])
    boot_auc = group_bootstrap(y, scores, g, roc_auc_score, n_boot, cfg["seed"])

    rows = []
    for n in names:
        p = scores[n]
        row = {"model": n, "selected": n == selected,
               "average_precision": average_precision_score(y, p),
               "ap_ci_low": ci(boot_ap[n])[0], "ap_ci_high": ci(boot_ap[n])[1],
               "roc_auc": roc_auc_score(y, p),
               "auc_ci_low": ci(boot_auc[n])[0], "auc_ci_high": ci(boot_auc[n])[1],
               "brier": brier_score_loss(y, p)}
        for mp in cfg["evaluation"]["fixed_precisions"]:
            k = int(mp * 100)
            _, thr = recall_at_precision(val["label"].to_numpy(), val[n].to_numpy(), mp)
            if np.isnan(thr):           # val never reaches that precision
                row.update({f"thr_p{k}_from_val": thr, f"test_precision_p{k}": np.nan,
                            f"test_recall_p{k}": 0.0})
                continue
            c = confusion_at(y, p, thr)
            row.update({f"thr_p{k}_from_val": thr, f"test_precision_p{k}": c["precision"],
                        f"test_recall_p{k}": c["recall"]})
        rows.append(row)
    table = pd.DataFrame(rows)

    # Paired AP differences: selected model minus each other model, same resamples.
    paired = {}
    for n in names:
        if n == selected:
            continue
        d = boot_ap[selected] - boot_ap[n]
        paired[n] = {"ap_diff": float(table.set_index("model").loc[selected, "average_precision"]
                                      - table.set_index("model").loc[n, "average_precision"]),
                     "ci": ci(d), "frac_boot_diff_le_0": float(np.mean(d <= 0))}
    extra = {"n_test": int(len(y)), "n_test_defects": int(y.sum()),
             "n_test_groups": int(len(np.unique(g))),
             "n_boot_valid": int(len(boot_ap[names[0]])), "selected_model": selected,
             "paired_ap_vs_selected": paired}
    return table, extra


def operating_point(val, test, model, cfg) -> dict:
    """Confusion on test at the val threshold for the strictest fixed precision."""
    mp = max(cfg["evaluation"]["fixed_precisions"])
    _, thr = recall_at_precision(val["label"].to_numpy(), val[model].to_numpy(), mp)
    source = f"val threshold for {mp:.0%} precision"
    if np.isnan(thr):
        thr, source = 0.5, "0.5 (val never reached target precision)"
    c = confusion_at(test["label"].to_numpy(), test[model].to_numpy(), thr)
    c["threshold_source"] = source
    # Are the errors the possibly-mislabelled images from Phase 1?
    pred = test[model].to_numpy() >= thr
    err = pred != test["label"].astype(bool).to_numpy()
    conf = test["label_conflict"].astype(bool).to_numpy()
    c["errors_with_label_conflict"] = int((err & conf).sum())
    c["label_conflict_images"] = int(conf.sum())
    c["error_rate_label_conflict"] = float(err[conf].mean()) if conf.any() else float("nan")
    c["error_rate_other"] = float(err[~conf].mean())
    return c


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def fig_curves(test, names, selected, cfg):
    y = test["label"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for n in names:
        lw, ls = (2.6, "-") if n == selected else (1.5, "--" if n == BASELINE else "-")
        prec, rec, _ = precision_recall_curve(y, test[n])
        axes[0].step(rec, prec, where="post", color=COLORS[n], lw=lw, ls=ls,
                     label=f"{n} (AP {average_precision_score(y, test[n]):.3f})")
        fpr, tpr, _ = roc_curve(y, test[n])
        axes[1].plot(fpr, tpr, color=COLORS[n], lw=lw, ls=ls,
                     label=f"{n} (AUC {roc_auc_score(y, test[n]):.3f})")
    axes[0].axhline(y.mean(), color=INK_2, lw=1, ls=":")
    axes[0].annotate(f"chance = prevalence {y.mean():.2f}", (0.01, y.mean()), xytext=(0, 4),
                     textcoords="offset points", fontsize=8, color=INK_2)
    axes[0].set(xlabel="Recall (defects caught)", ylabel="Precision", ylim=(0, 1.03),
                title="Precision-recall (test)")
    axes[1].plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":")
    axes[1].set(xlabel="False positive rate (good parts rejected)",
                ylabel="True positive rate", title="ROC (test)")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout()
    save_fig(fig, tagged(cfg, "pr_roc_curves"), cfg)


def fig_reliability(test, names, cfg):
    y = test["label"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":", label="perfectly calibrated")
    for n in names:
        frac, mean_p = calibration_curve(y, test[n], n_bins=8, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=4, color=COLORS[n],
                ls="--" if n == BASELINE else "-", lw=1.5, label=n)
    ax.set(xlabel="Predicted defect probability", ylabel="Observed defect fraction",
           title="Reliability (test, quantile bins)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7.5)
    save_fig(fig, tagged(cfg, "reliability"), cfg)


def fig_confusion(op, model, cfg):
    m = np.array([[op["tn"], op["fp"]], [op["fn"], op["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = m[i, j] / m[i].sum()
            ax.text(j, i, f"{m[i, j]}\n({share:.0%} of row)", ha="center", va="center",
                    color="white" if share > 0.6 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["pred OK", "pred Defect"])
    ax.set_yticks([0, 1], ["true OK", "true Defect"])
    ax.grid(False)
    ax.set_title(f"{model} @ thr {op['threshold']:.3f}\n({op['threshold_source']})",
                 fontsize=10)
    save_fig(fig, tagged(cfg, "confusion_matrix"), cfg)


def fig_errors(test, model, thr, cfg, n=6):
    """Hardest cases: defects scored lowest and OK parts scored highest.

    Always the n hardest of each class (errors first by construction), so the
    figure shows the decision boundary even when the model makes few mistakes.
    """
    fn = test[test["label"] == 1].nsmallest(n, model)
    fp = test[test["label"] == 0].nlargest(n, model)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.9), squeeze=False)
    for r, (rows, title, lab) in enumerate([(fn, "Defects scored lowest", 1),
                                            (fp, "OK parts scored highest", 0)]):
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            if c >= len(rows):
                ax.axis("off")
                continue
            row = rows.iloc[c]
            ax.imshow(Image.open(resolve(row["path"])).convert("L"), cmap="gray")
            wrong = (row[model] >= thr) != bool(row["label"])
            flag = " · label?" if row["label_conflict"] else ""
            ax.set_title(f"p={row[model]:.3f}{flag}", fontsize=8, fontweight="normal",
                         color=PALETTE[7] if wrong else INK_2)
            for s in ax.spines.values():
                s.set_visible(True), s.set_color(CLASS_COLORS[lab]), s.set_linewidth(2)
        axes[r, 0].set_ylabel(title, fontsize=9, color=CLASS_COLORS[lab])
    fig.suptitle(f"{model}: hardest test images (red title = wrong at thr {thr:.3f})",
                 fontweight="bold")
    save_fig(fig, tagged(cfg, "error_gallery"), cfg)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cfg = apply_fast(load_config(), args.fast)
    set_seed(cfg["seed"])
    plot_style()
    device = get_device()
    metrics_dir = get_path(cfg, "metrics_dir")

    comp_path = metrics_dir / f"{tagged(cfg, 'model_comparison')}.csv"
    if not comp_path.exists():
        sys.exit(f"{comp_path.name} not found - run scripts/03_train_models.py"
                 f"{' --fast' if args.fast else ''} first.")
    comp = pd.read_csv(comp_path)

    print("Predicting val + test ...")
    val, test, found = collect_predictions(cfg, device)
    comp = comp[comp["model"].isin(found)]
    selected = comp.loc[comp["average_precision"].idxmax(), "model"]
    names = found + [BASELINE]
    print(f"Selected model (best val AP): {selected}")

    val.to_csv(metrics_dir / f"{tagged(cfg, 'predictions_val')}.csv", index=False)
    test.to_csv(metrics_dir / f"{tagged(cfg, 'predictions_test')}.csv", index=False)

    table, extra = test_metrics(val, test, names, selected, cfg, args.n_boot)
    op = operating_point(val, test, selected, cfg)
    table.to_csv(metrics_dir / f"{tagged(cfg, 'test_metrics')}.csv", index=False)
    save_json({**extra, "operating_point_selected": op,
               "metrics": table.to_dict("records")},
              metrics_dir / f"{tagged(cfg, 'test_metrics')}.json")

    fig_curves(test, names, selected, cfg)
    fig_reliability(test, names, cfg)
    fig_confusion(op, selected, cfg)
    fig_errors(test, selected, op["threshold"], cfg)

    print(f"\nTest set: {extra['n_test']} images, {extra['n_test_defects']} defects, "
          f"{extra['n_test_groups']} duplicate groups; {extra['n_boot_valid']} bootstrap resamples")
    for _, r in table.iterrows():
        print(f"  {r['model']:<20} AP {r['average_precision']:.3f} "
              f"[{r['ap_ci_low']:.3f}, {r['ap_ci_high']:.3f}]  "
              f"AUC {r['roc_auc']:.3f} [{r['auc_ci_low']:.3f}, {r['auc_ci_high']:.3f}]  "
              f"P95-thr -> test P {r['test_precision_p95']:.3f} R {r['test_recall_p95']:.3f}"
              f"{'  <- selected' if r['selected'] else ''}")
    print(f"\nPaired AP difference ({selected} minus other):")
    for n, d in extra["paired_ap_vs_selected"].items():
        print(f"  vs {n:<20} {d['ap_diff']:+.3f}  95% CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    print(f"\nOperating point ({op['threshold_source']}, thr {op['threshold']:.3f}): "
          f"TP {op['tp']} FP {op['fp']} FN {op['fn']} TN {op['tn']}  -> precision "
          f"{op['precision']:.3f}, recall {op['recall']:.3f}, "
          f"false rejects {op['false_reject_rate']:.2%}")
    print(f"Label-conflict images: {op['label_conflict_images']}, "
          f"of which wrong: {op['errors_with_label_conflict']}")


if __name__ == "__main__":
    main()
