"""Phase 5: choose the decision threshold from asymmetric costs.

A shipped defect (missed) costs `cost_escaped_defect`; rejecting a good part costs
`cost_false_reject`. The threshold that minimises expected cost per part on the
VALIDATION split is applied unchanged to test - and once, as the final check, to
the untouched production split. The cost curve is smoothed (kernel density of
each class's scores) so the choice allows for defects scoring lower than any
seen in validation; the raw empirical minimum is reported for comparison.

Compared against:
  * the empirical (unsmoothed) cost minimum on val,
  * 0.5 (the default people reach for),
  * the 95%-precision threshold from Phase 4,
  * the textbook Bayes threshold c_fp / (c_fp + c_fn) after correcting the
    probabilities for the sampler's ~50% training prior,
  * no model at all: ship everything / reject everything.
Plus: every model at its own cost-optimal threshold (models compared in cost
terms), and how the threshold moves when the cost ratio is 5, 20, 50 or 100.

Inputs: results/metrics/predictions_{val,test}.csv and test_metrics.json (Phase 4)
Outputs (suffix _fast in --fast mode):
  results/metrics/threshold.json (used by the demo), threshold_candidates.csv,
  cost_sensitivity.csv, model_costs.csv
  results/figures/cost_curve.png, cost_sensitivity.png, model_costs.png

Run:  python scripts/05_threshold.py [--fast] [--n-boot 2000]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import make_loaders  # noqa: E402
from src.evaluation import (choose_threshold, ci, confusion_at, cost_curve,  # noqa: E402
                            expected_cost, group_bootstrap, predict, recall_at_precision,
                            smoothed_cost_curve)
from src.models import MODEL_NAMES, load_checkpoint  # noqa: E402
from src.utils import (INK_2, PALETTE, apply_fast, get_device, get_path, load_config,  # noqa: E402
                       plot_style, save_fig, save_json, set_seed, tagged)

BASELINE = "background_only"
COLORS = {**{m: PALETTE[i] for i, m in enumerate(MODEL_NAMES)}, BASELINE: "#8a8984"}
PRIMARY = "cost-optimal, smoothed (val)"
CAND_COLORS = {PRIMARY: PALETTE[2], "cost-optimal, empirical (val)": PALETTE[4],
               "0.5 default": PALETTE[7],
               "95% precision (val)": PALETTE[6], "Bayes + prior correction": PALETTE[3]}


def train_prior(cfg) -> float:
    """Defect share the model saw in training (sampler ~0.5, else the train share)."""
    p = get_path(cfg, "metrics_dir") / "splits_summary.json"
    if p.exists():
        s = json.loads(p.read_text())["sampler"]
        if cfg["train"]["imbalance"] == "sampler":
            return s["defect_frac_served_per_epoch"]
        return s["train_defect_frac"]
    return 0.5


def bayes_raw_threshold(c_fn, c_fp, pi_train, pi_line) -> float:
    """Raw-probability threshold equal to c_fp/(c_fp+c_fn) on prior-corrected probabilities."""
    t = c_fp / (c_fp + c_fn)
    odds = t / (1 - t) * (pi_train / (1 - pi_train)) / (pi_line / (1 - pi_line))
    return float(odds / (1 + odds))


def summarize(y, p, thr, c_fn, c_fp, pi) -> dict:
    c = confusion_at(y, p, thr)
    cost = expected_cost(y, p, thr, c_fn, c_fp, pi)
    miss = 1 - c["recall"]
    return {"threshold": thr, "cost_per_part": cost, "cost_per_1000": 1000 * cost,
            "recall": c["recall"], "precision": c["precision"],
            "false_reject_rate": c["false_reject_rate"],
            # Per 1000 parts on a line with prevalence pi:
            "escapes_per_1000": 1000 * pi * miss,
            "false_rejects_per_1000": 1000 * (1 - pi) * c["false_reject_rate"],
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"]}


def boot_cost(df, model, thr, c_fn, c_fp, pi, n_boot, seed):
    y, g = df["label"].to_numpy(), df["dup_group"].to_numpy()
    b = group_bootstrap(y, {model: df[model].to_numpy()}, g,
                        lambda yb, sb: expected_cost(yb, sb, thr, c_fn, c_fp, pi), n_boot, seed)
    return ci(b[model] * 1000)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def fig_cost_curve(curves, cands, model, ship_all, reject_all, cfg):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for split, ls, lw, label in (("val", ":", 1.4, "val (empirical)"),
                                 ("val_smoothed", "-", 2.2, "val (smoothed - used)"),
                                 ("test", "--", 1.4, "test (empirical)")):
        c = curves[split]
        ax.plot(c["threshold"], 1000 * c["cost"], color=COLORS[model], ls=ls, lw=lw,
                label=f"{model} - {label}")
    ax.axhline(1000 * reject_all, color=INK_2, lw=1, ls=":")
    ax.annotate(f"reject everything: {1000 * reject_all:.0f}", (0.99, 1000 * reject_all),
                xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color=INK_2)
    for name, row in cands.iterrows():
        ax.axvline(row["threshold"], color=CAND_COLORS[name], lw=1.2, alpha=0.9)
        ax.plot(row["threshold"], row["test_cost_per_1000"], "o", color=CAND_COLORS[name],
                ms=6, label=f"{name}: t={row['threshold']:.3f}")
    ax.set_ylim(0, 1000 * reject_all * 1.15)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Threshold (call 'defect' when p ≥ t)")
    ax.set_ylabel("Expected cost per 1000 parts")
    ax.set_title(f"Cost curve - ship-everything costs {1000 * ship_all:.0f} (off chart)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save_fig(fig, tagged(cfg, "cost_curve"), cfg)


def fig_sensitivity(sens, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = np.arange(len(sens))
    labels = [f"{r:g}:1" for r in sens["cost_ratio"]]
    axes[0].plot(x, sens["threshold"], marker="o", color=PALETTE[2])
    for xi, t in zip(x, sens["threshold"]):
        axes[0].annotate(f"{t:.3f}", (xi, t), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=8, color=INK_2)
    axes[0].set(title="Cost-optimal threshold (smoothed val curve)", ylabel="Threshold")
    axes[1].plot(x, sens["escapes_per_1000"], marker="o", color=PALETTE[1],
                 label="defects shipped")
    axes[1].plot(x, sens["false_rejects_per_1000"], marker="o", color=PALETTE[0],
                 label="good parts rejected")
    axes[1].set(title="Outcome on test, per 1000 parts", ylabel="Parts per 1000")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Cost of a shipped defect : cost of a false reject")
    fig.tight_layout()
    save_fig(fig, tagged(cfg, "cost_sensitivity"), cfg)


def fig_model_costs(mc, ship_all, reject_all, cfg):
    mc = mc.sort_values("test_cost_per_1000")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(mc))
    err = np.vstack([mc["test_cost_per_1000"] - mc["ci_low"], mc["ci_high"] - mc["test_cost_per_1000"]])
    ax.barh(y, mc["test_cost_per_1000"], xerr=err, color=[COLORS[m] for m in mc["model"]],
            error_kw={"ecolor": INK_2, "lw": 1, "capsize": 3})
    for yi, v in zip(y, mc["test_cost_per_1000"]):
        # A model whose best option is "reject everything" adds nothing over no model.
        note = "  (= reject everything)" if np.isclose(v, 1000 * reject_all) else ""
        if note:   # label inside the bar, clear of the CI cap
            ax.annotate(f"{v:.1f}{note}", (v, yi), xytext=(-10, 0), textcoords="offset points",
                        fontsize=8.5, color="white", ha="right", va="center")
        else:
            ax.annotate(f"{v:.1f}", (v, yi), xytext=(4, 6), textcoords="offset points",
                        fontsize=8.5, color=INK_2)
    ax.set_yticks(y, mc["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Test cost per 1000 parts (each model at its own val-chosen threshold, 95% CI)")
    ax.set_title(f"Models in cost terms (shipping everything = {1000 * ship_all:.0f})")
    ax.grid(axis="y", visible=False)
    save_fig(fig, tagged(cfg, "model_costs"), cfg)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cfg = apply_fast(load_config(), args.fast)
    set_seed(cfg["seed"])
    plot_style()
    metrics_dir = get_path(cfg, "metrics_dir")
    ev = cfg["evaluation"]
    c_fn, c_fp = ev["cost_escaped_defect"], ev["cost_false_reject"]
    pi = cfg["data"]["defect_ratio"]
    grid = np.linspace(0, 1, ev["threshold_steps"])

    try:
        val = pd.read_csv(metrics_dir / f"{tagged(cfg, 'predictions_val')}.csv")
        test = pd.read_csv(metrics_dir / f"{tagged(cfg, 'predictions_test')}.csv")
        tm = json.loads((metrics_dir / f"{tagged(cfg, 'test_metrics')}.json").read_text())
    except FileNotFoundError as e:
        sys.exit(f"{e.filename} not found - run scripts/04_evaluate.py"
                 f"{' --fast' if args.fast else ''} first.")
    model = tm["selected_model"]
    models = [m for m in MODEL_NAMES + [BASELINE] if m in val.columns]
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    ship_all, reject_all = c_fn * pi, c_fp * (1 - pi)
    print(f"Model {model} | cost ratio {c_fn}:{c_fp} | line prevalence {pi:.0%}")

    # 1. Candidate thresholds for the selected model ----------------------------
    curves = {s: cost_curve(d["label"].to_numpy(), d[model].to_numpy(), grid, c_fn, c_fp, pi)
              for s, d in (("val", val), ("test", test))}
    curves["val_smoothed"] = smoothed_cost_curve(yv, val[model].to_numpy(), grid, c_fn, c_fp, pi)
    pi_tr = train_prior(cfg)
    _, t95 = recall_at_precision(yv, val[model].to_numpy(), max(ev["fixed_precisions"]))
    cand_thr = {PRIMARY: choose_threshold(curves["val_smoothed"]),
                "cost-optimal, empirical (val)": choose_threshold(curves["val"]),
                "0.5 default": 0.5,
                "95% precision (val)": t95 if not np.isnan(t95) else 0.5,
                "Bayes + prior correction": bayes_raw_threshold(c_fn, c_fp, pi_tr, pi)}
    rows = []
    for name, t in cand_thr.items():
        v = summarize(yv, val[model].to_numpy(), t, c_fn, c_fp, pi)
        s = summarize(yt, test[model].to_numpy(), t, c_fn, c_fp, pi)
        lo, hi = boot_cost(test, model, t, c_fn, c_fp, pi, args.n_boot, cfg["seed"])
        rows.append({"candidate": name, "threshold": t, "val_cost_per_1000": v["cost_per_1000"],
                     "test_cost_per_1000": s["cost_per_1000"], "test_cost_ci_low": lo,
                     "test_cost_ci_high": hi, **{f"test_{k}": s[k] for k in
                     ("recall", "precision", "false_reject_rate", "escapes_per_1000",
                      "false_rejects_per_1000", "tp", "fp", "fn", "tn")}})
    cands = pd.DataFrame(rows).set_index("candidate")
    cands.to_csv(metrics_dir / f"{tagged(cfg, 'threshold_candidates')}.csv")
    t_star = cand_thr[PRIMARY]

    # 2. Every model at its own cost-optimal threshold ------------------------
    mrows = []
    for m in models:
        cv = smoothed_cost_curve(yv, val[m].to_numpy(), grid, c_fn, c_fp, pi)
        t = choose_threshold(cv)
        s = summarize(yt, test[m].to_numpy(), t, c_fn, c_fp, pi)
        lo, hi = boot_cost(test, m, t, c_fn, c_fp, pi, args.n_boot, cfg["seed"])
        mrows.append({"model": m, "threshold": t, "test_cost_per_1000": s["cost_per_1000"],
                      "ci_low": lo, "ci_high": hi, "test_recall": s["recall"],
                      "test_false_reject_rate": s["false_reject_rate"]})
    mc = pd.DataFrame(mrows)
    mc.to_csv(metrics_dir / f"{tagged(cfg, 'model_costs')}.csv", index=False)

    # 3. Sensitivity to the cost ratio ---------------------------------------
    srows = []
    for r in ev["cost_ratios_sensitivity"]:
        cv = smoothed_cost_curve(yv, val[model].to_numpy(), grid, r * c_fp, c_fp, pi)
        t = choose_threshold(cv)
        s = summarize(yt, test[model].to_numpy(), t, r * c_fp, c_fp, pi)
        srows.append({"cost_ratio": r, "threshold": t, **{k: s[k] for k in
                      ("cost_per_1000", "recall", "false_reject_rate", "escapes_per_1000",
                       "false_rejects_per_1000")}})
    sens = pd.DataFrame(srows)
    sens.to_csv(metrics_dir / f"{tagged(cfg, 'cost_sensitivity')}.csv", index=False)

    # 4. Final check on the untouched production split (its only use) --------
    device = get_device()
    ckpt = get_path(cfg, "models_dir") / f"{tagged(cfg, model)}.pt"
    prod = None
    if model != BASELINE and ckpt.exists():
        net, _ = load_checkpoint(ckpt, device)
        p, y, _ = predict(net, make_loaders(cfg, splits=("production",))["production"], device)
        prod = summarize(y, p, t_star, c_fn, c_fp, pi)
        prod["n"], prod["n_defect"] = int(len(y)), int(y.sum())

    out = {"model": model, "threshold": t_star, "cost_escaped_defect": c_fn,
           "cost_false_reject": c_fp, "line_prevalence": pi, "train_prior": pi_tr,
           "image_size": cfg["image"]["size"], "is_fast": cfg["is_fast"],
           "chosen_on": "val (smoothed cost curve)", "val": summarize(yv, val[model].to_numpy(), t_star, c_fn, c_fp, pi),
           "test": summarize(yt, test[model].to_numpy(), t_star, c_fn, c_fp, pi),
           "test_cost_per_1000_ci": list(cands.loc[PRIMARY,
                                                   ["test_cost_ci_low", "test_cost_ci_high"]]),
           "production": prod,
           "reference_cost_per_1000": {"ship_everything": 1000 * ship_all,
                                       "reject_everything": 1000 * reject_all}}
    save_json(out, metrics_dir / f"{tagged(cfg, 'threshold')}.json")

    fig_cost_curve(curves, cands, model, ship_all, reject_all, cfg)
    fig_sensitivity(sens, cfg)
    fig_model_costs(mc, ship_all, reject_all, cfg)

    # Console summary ---------------------------------------------------------
    print(f"\nReference: ship everything {1000 * ship_all:.0f} / 1000 parts, "
          f"reject everything {1000 * reject_all:.0f} / 1000 parts")
    print(f"\nCandidate thresholds ({model}, cost per 1000 parts):")
    for name, r in cands.iterrows():
        print(f"  {name:<26} t={r['threshold']:.4f}  val {r['val_cost_per_1000']:6.1f}  "
              f"test {r['test_cost_per_1000']:6.1f} [{r['test_cost_ci_low']:.1f}, "
              f"{r['test_cost_ci_high']:.1f}]  recall {r['test_recall']:.3f}  "
              f"FRR {r['test_false_reject_rate']:.3f}")
    print("\nModels at their own cost-optimal threshold (test cost per 1000):")
    for _, r in mc.sort_values("test_cost_per_1000").iterrows():
        print(f"  {r['model']:<20} t={r['threshold']:.4f}  {r['test_cost_per_1000']:6.1f} "
              f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]")
    print("\nCost-ratio sensitivity (test):")
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if prod:
        print(f"\nProduction check ({prod['n']} parts, {prod['n_defect']} defects) at t={t_star:.4f}: "
              f"cost {prod['cost_per_1000']:.1f}/1000, recall {prod['recall']:.3f}, "
              f"FRR {prod['false_reject_rate']:.3f}  (TP {prod['tp']} FP {prod['fp']} "
              f"FN {prod['fn']} TN {prod['tn']})")


if __name__ == "__main__":
    main()
