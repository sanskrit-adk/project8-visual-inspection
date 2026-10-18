"""Phase 10: write report/REPORT_NOTES.md from the saved metrics.

Every number in the notes is read from results/metrics/ - nothing is typed by
hand - so the report can't drift from the code. Full-run results are used when
present; otherwise the --fast ones, and every section says which it used.

Run:  python scripts/10_make_report_notes.py
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.utils import ROOT, get_path, load_config  # noqa: E402

CFG = load_config()
M = get_path(CFG, "metrics_dir")
FIG = get_path(CFG, "figures_dir")


def load(name):
    """(data, source) preferring the full run; data is None if neither exists."""
    for suffix, src in (("", "full run"), ("_fast", "FAST smoke run")):
        for ext in (".json", ".csv"):
            p = M / f"{name}{suffix}{ext}"
            if p.exists():
                data = json.loads(p.read_text()) if ext == ".json" else pd.read_csv(p)
                return data, src, suffix
    return None, "missing", ""


def fig(name, suffix=""):
    for s in (suffix, "", "_fast"):
        p = FIG / f"{name}{s}.png"
        if p.exists():
            return f"`{p.relative_to(ROOT).as_posix()}`"
    return f"`results/figures/{name}.png` (not generated)"


def f3(v):
    return "n/a" if v is None or v != v else f"{v:.3f}"


def table(df, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def main():
    out = [f"# Report notes - Project 8: Automated Visual Quality Inspection\n",
           f"_Generated {date.today().isoformat()} by `scripts/10_make_report_notes.py`. "
           "Every number below is read from `results/metrics/`. Sections marked "
           "**FAST smoke run** come from 128px / 3-epoch / 25%-data runs and must be "
           "regenerated from the full run before submission._\n"]

    env, _, _ = load("environment")
    if env:
        out.append(f"**Environment:** Python {env['python']}, torch {env['packages']['torch']}, "
                   f"device `{env['device']}` ({env['cpu_threads']} CPU threads), seed {env['seed']}.\n")

    # Phase 1 --------------------------------------------------------------
    eda, src, _ = load("eda_summary")
    sb, _, _ = load("shortcut_baseline")
    if eda:
        d = eda["duplicates"]
        out.append("## 1. Data and exploration\n")
        out.append(f"- {eda['n_images']} labelled images, all "
                   f"{eda['image_sizes'][0]['width']}x{eda['image_sizes'][0]['height']}; raw defect "
                   f"share {eda['defect_fraction_raw']:.1%} (counts: {eda['counts']}).")
        out.append(f"- Near-duplicates (thumbnail correlation >= {d['threshold']:.3f}): "
                   f"{d['images_in_dup_groups']} images ({d['frac_images_in_dup_groups']:.0%}) in "
                   f"{d['n_groups_size_gt1']} groups (largest {d['largest_group']}); "
                   f"**{d['groups_spanning_orig_train_and_test']} groups span Kaggle's train and test "
                   f"folders** -> the published split leaks.")
        out.append(f"- {d['images_whose_nn_copy_has_other_label']} images have a near-copy with the "
                   f"other label (possible label noise).")
        if sb:
            out.append(f"- Shortcut baseline (logistic regression, group 5-fold CV): background-only "
                       f"ROC-AUC **{sb['background_only']['roc_auc']:.3f}**, all simple stats "
                       f"{sb['all_stats']['roc_auc']:.3f}. OK and defect photos differ in lighting.")
        out.append(f"- Figures: {fig('eda_samples')}, {fig('eda_stat_distributions')}, "
                   f"{fig('eda_duplicate_examples')}, {fig('eda_cross_label_duplicates')}, "
                   f"{fig('eda_shortcut_baseline')}\n")

    # Phase 2 --------------------------------------------------------------
    sp, _, _ = load("splits_summary")
    if sp:
        out.append("## 2. Splits\n")
        rows = pd.DataFrame([{"split": k, **v} for k, v in sp["splits"].items()])
        out.append(table(rows, ["split", "n", "n_ok", "n_defect", "defect_frac", "n_defect_groups"],
                         {"defect_frac": lambda v: f"{v:.1%}"}))
        sc = sp["shortcut_train_to_test"]
        out.append(f"- Split by duplicate group; leakage check: "
                   f"{sp['leakage']['duplicate_groups_shared_between_splits']} shared groups, "
                   f"{sp['leakage']['files_shared_between_splits']} shared files.")
        out.append(f"- Background-only baseline on the final test split: ROC-AUC "
                   f"**{sc['background_only']['roc_auc']:.3f}**, AP {sc['background_only']['average_precision']:.3f} "
                   f"(prevalence {sc['test_prevalence']:.2f}).")
        out.append(f"- Sampler serves {sp['sampler']['defect_frac_served_per_epoch']:.0%} defects per epoch. "
                   f"Figures: {fig('splits_composition')}, {fig('augmentation_examples')}\n")

    # Phase 3 --------------------------------------------------------------
    mc, src, sfx = load("model_comparison")
    if mc is not None:
        out.append(f"## 3. Models (validation) - {src}\n")
        out.append(table(mc, ["model", "average_precision", "roc_auc", "recall_at_p95", "best_epoch",
                              "train_sec", "params_trainable"],
                         {"train_sec": lambda v: f"{v:.0f}", "params_trainable": lambda v: f"{v:,}"}))
        hist, _, _ = load("training_history")
        if hist:
            out.append(f"- Background-only baseline on val: AP "
                       f"{hist['background_only_val']['average_precision']:.3f}.")
        sc, ssrc, _ = load("data_scarcity")
        if sc is not None:
            piv = sc.pivot(index="n_defect", columns="model", values="average_precision").reset_index()
            out.append(f"\nData scarcity (val AP by number of training defects, {ssrc}):\n")
            out.append(table(piv, list(piv.columns), {"n_defect": lambda v: str(int(v))}))
        out.append(f"Figures: {fig('model_comparison', sfx)}, {fig('training_curves', sfx)}, "
                   f"{fig('data_scarcity', sfx)}\n")

    # Phase 4 --------------------------------------------------------------
    tm, src, sfx = load("test_metrics")
    if tm:
        out.append(f"## 4. Test evaluation - {src}\n")
        out.append(f"Test: {tm['n_test']} images, {tm['n_test_defects']} defects, "
                   f"{tm['n_test_groups']} duplicate groups. Selected model (best val AP): "
                   f"**{tm['selected_model']}**. 95% CIs from a group bootstrap.\n")
        t = pd.DataFrame(tm["metrics"])
        t["AP [95% CI]"] = t.apply(lambda r: f"{r['average_precision']:.3f} [{r['ap_ci_low']:.3f}, "
                                             f"{r['ap_ci_high']:.3f}]", axis=1)
        t["AUC [95% CI]"] = t.apply(lambda r: f"{r['roc_auc']:.3f} [{r['auc_ci_low']:.3f}, "
                                              f"{r['auc_ci_high']:.3f}]", axis=1)
        out.append(table(t, ["model", "AP [95% CI]", "AUC [95% CI]", "brier", "test_precision_p95",
                             "test_recall_p95"], {"test_precision_p95": f3, "test_recall_p95": f3}))
        out.append("Paired AP difference (selected minus other):\n")
        for n, d in tm["paired_ap_vs_selected"].items():
            sig = "" if d["ci"][0] <= 0 <= d["ci"][1] else " (CI excludes 0)"
            out.append(f"- vs {n}: {d['ap_diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]{sig}")
        op = tm["operating_point_selected"]
        out.append(f"\nAt the val 95%-precision threshold ({op['threshold']:.3f}): TP {op['tp']}, "
                   f"FP {op['fp']}, FN {op['fn']}, TN {op['tn']}.")
        out.append(f"Figures: {fig('pr_roc_curves', sfx)}, {fig('reliability', sfx)}, "
                   f"{fig('confusion_matrix', sfx)}, {fig('error_gallery', sfx)}\n")
    neu, src, sfx = load("neu_per_type")
    if neu:
        out.append(f"### 4b. Defect types (NEU-DET) - {src}\n")
        out.append(f"ResNet18 fine-tuned, 6 classes: accuracy **{neu['accuracy']:.3f}**, macro-F1 "
                   f"{neu['macro_f1']:.3f} on {neu['n_test']} test images. Most confused: "
                   f"{neu['most_confused_pair'][0]} <-> {neu['most_confused_pair'][1]} "
                   f"({neu['most_confused_pair'][2]} images).\n")
        pc = pd.DataFrame([{"type": k, **v} for k, v in neu["per_class"].items()])
        out.append(table(pc, ["type", "precision", "recall", "f1-score"]))
        out.append(f"Figure: {fig('neu_confusion_matrix', sfx)}\n")

    # Phase 5 --------------------------------------------------------------
    th, src, sfx = load("threshold")
    if th:
        out.append(f"## 5. Cost-based threshold - {src}\n")
        out.append(f"Costs: shipped defect {th['cost_escaped_defect']}, false reject "
                   f"{th['cost_false_reject']}, line prevalence {th['line_prevalence']:.0%}. "
                   f"Threshold **{th['threshold']:.3f}** for {th['model']} (chosen on "
                   f"{th['chosen_on']}).\n")
        ref = th["reference_cost_per_1000"]
        lo, hi = th["test_cost_per_1000_ci"]
        out.append(f"- Test: cost **{th['test']['cost_per_1000']:.1f}** per 1000 parts "
                   f"[{lo:.1f}, {hi:.1f}], recall {th['test']['recall']:.3f}, false-reject rate "
                   f"{th['test']['false_reject_rate']:.1%}. Ship-everything costs "
                   f"{ref['ship_everything']:.0f}; reject-everything {ref['reject_everything']:.0f}.")
        if th.get("production"):
            pr = th["production"]
            out.append(f"- Production split (used once): cost {pr['cost_per_1000']:.1f} per 1000, "
                       f"recall {pr['recall']:.3f}, false-reject rate {pr['false_reject_rate']:.1%} "
                       f"({pr['n']} parts, {pr['n_defect']} defects).")
        cands, _, _ = load("threshold_candidates")
        if cands is not None:
            out.append("\n" + table(cands, ["candidate", "threshold", "test_cost_per_1000",
                                            "test_recall", "test_false_reject_rate"]))
        out.append(f"Figures: {fig('cost_curve', sfx)}, {fig('cost_sensitivity', sfx)}, "
                   f"{fig('model_costs', sfx)}\n")

    # Phase 6 --------------------------------------------------------------
    lat, src, sfx = load("latency")
    if lat:
        out.append(f"## 6. Latency - {src}\n")
        df = pd.DataFrame(lat["results"])
        out.append(f"Device `{lat['device']}`, {lat['cpu_threads']} threads, {lat['image_size']}px; "
                   f"preprocessing {lat['preprocess_ms_median']:.1f} ms/image; line budget "
                   f"{lat['line_budget_ms_per_part']:.0f} ms/part.\n")
        out.append(table(df, ["model", "batch_size", "median_ms_per_image", "p95_ms_per_image",
                              "images_per_sec", "headroom_vs_line", "checkpoint_mb"],
                         {"headroom_vs_line": lambda v: f"{v:.0f}x",
                          "batch_size": lambda v: str(int(v))}))
        out.append(f"Figure: {fig('latency', sfx)}\n")

    # Phase 7 --------------------------------------------------------------
    gc, src, sfx = load("gradcam")
    if gc:
        c = gc["castings"]
        out.append(f"## 7. Grad-CAM - {src}\n")
        out.append(f"- Castings ({c['model']}, images flagged as defect, n={c['flagged']['n']}): "
                   f"{c['flagged']['part_energy']:.0%} of heatmap mass on the part (part = "
                   f"{c['part_area_frac']:.0%} of the image; score **{c['sanity_score_part']:.2f}**), "
                   f"{c['flagged']['bg_energy']:.0%} on the background ring "
                   f"({c['bg_area_frac']:.0%} of the image; score {c['sanity_score_bg']:.2f}). "
                   "Score > 1 = preference for that region.")
        out.append(f"- Randomisation check: Spearman correlation between trained and random-weight "
                   f"maps = {c['randomization_spearman_mean']:.3f} (low = maps depend on what was learned).")
        if gc.get("neu"):
            n = gc["neu"]
            out.append(f"- NEU: pointing game **{n['pointing_game_acc']:.0%}** (peak inside an annotated "
                       f"box); {n['box_energy_mean']:.0%} of mass inside boxes that cover "
                       f"{n['box_area_mean']:.0%} of the image.")
        out.append(f"- Figures: {fig('gradcam_castings', sfx)}, {fig('gradcam_part_vs_background', sfx)}, "
                   f"{fig('gradcam_neu', sfx)}\n")

    # Phase 8 --------------------------------------------------------------
    rb, src, sfx = load("robustness")
    if rb:
        out.append(f"## 8. Robustness and shortcut tests - {src}\n")
        sc = pd.DataFrame(rb["shortcut_ap"]).T.reset_index().rename(columns={"index": "model"})
        cols = ["model", "original", "part_only", "background_only", "background_equalized"]
        out.append(table(sc[cols], cols))
        out.append("Reading: a model using the part keeps its AP on *part only* and *background "
                   "equalised*, and falls to chance on *background only*.\n")
        rows = pd.DataFrame(rb["rows"])
        d = rows[(rows["model"] == rb["deployed_model"]) & rows["recall_at_thr"].notna()]
        out.append(f"Deployed model ({rb['deployed_model']}) at its fixed threshold "
                   f"{rb['threshold']:.3f} under imaging changes:\n")
        out.append(table(d, ["family", "level", "average_precision", "recall_at_thr", "frr_at_thr",
                             "cost_per_1000_at_thr"],
                         {"level": lambda v: f"{v:g}", "cost_per_1000_at_thr": lambda v: f"{v:.0f}"}))
        out.append(f"Figures: {fig('robustness_shift', sfx)}, {fig('shortcut_tests', sfx)}, "
                   f"{fig('perturbation_examples', sfx)}\n")

    out.append("## 9. Demo\n\n`streamlit run app/streamlit_app.py` - pick a held-out image or upload "
               "one; shows probability, accept/reject at the cost-based threshold, Grad-CAM, and a "
               "cost-ratio slider.\n")
    out.append("## Decision log\n\nSee `DECISIONS.md` (alternatives considered and evidence for each "
               "design choice).\n")

    path = get_path(CFG, "report_dir") / "REPORT_NOTES.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
