"""Phase 6: inference latency and throughput vs the line speed.

For every trained checkpoint: time the forward pass at batch size 1 (one part at a
time, as on a live line) and 32 (batched re-inspection), after warm-up runs, and
compare throughput with `latency.line_speed_parts_per_sec`. Also times the
preprocessing (decode + resize + normalise) so the end-to-end budget is honest.

Timings depend on the machine and on whatever else is running; the device, CPU
thread count and torch version are saved with the results.

Outputs (suffix _fast in --fast mode):
  results/metrics/latency.csv, latency.json, results/figures/latency.png

Run:  python scripts/06_latency.py [--fast]
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import build_transforms, load_split, resolve  # noqa: E402
from src.models import MODEL_NAMES, count_params, load_checkpoint  # noqa: E402
from src.utils import (INK_2, PALETTE, apply_fast, get_device, get_path, load_config,  # noqa: E402
                       plot_style, save_fig, save_json, set_seed, tagged)

COLORS = {m: PALETTE[i] for i, m in enumerate(MODEL_NAMES)}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_model(model, size, batch, warmup, runs, device) -> np.ndarray:
    """Seconds per forward pass for `runs` timed repetitions."""
    x = torch.randn(batch, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    times = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        times[i] = time.perf_counter() - t0
    return times


def time_preprocess(cfg, n=50) -> float:
    """Median seconds to open, decode, resize and normalise one image."""
    tf = build_transforms(cfg, train=False)
    paths = [resolve(p) for p in load_split(cfg, "test")["path"].head(n)]
    times = []
    for p in paths:
        t0 = time.perf_counter()
        tf(Image.open(p).convert("RGB"))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _log_ticks(ax, candidates):
    """Log x-axis with plain-number ticks inside the data range (no 2x10^1 clutter)."""
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ticks = [t for t in candidates if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def fig_latency(df, line_speed, cfg):
    d1 = df[df["batch_size"] == 1].set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    y = np.arange(len(d1))
    colors = [COLORS[m] for m in d1.index]
    err = np.vstack([d1["median_ms_per_image"] - d1["p5_ms_per_image"],
                     d1["p95_ms_per_image"] - d1["median_ms_per_image"]])
    axes[0].barh(y, d1["end_to_end_ms"], color=colors, alpha=0.35, label="+ preprocessing")
    axes[0].barh(y, d1["median_ms_per_image"], xerr=err, color=colors,
                 error_kw={"ecolor": INK_2, "lw": 1, "capsize": 3}, label="model forward")
    budget = 1000 / line_speed
    axes[0].axvline(budget, color=PALETTE[7], ls="--", lw=1.2)
    axes[0].annotate(f"line budget {budget:.0f} ms/part", (budget, -0.5),
                     xytext=(-4, 0), textcoords="offset points", ha="right", va="top",
                     fontsize=8, color=PALETTE[7])
    _log_ticks(axes[0], [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    axes[0].set_yticks(y, d1.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ms per image, batch 1 (log scale; bar = median, whiskers = p5-p95)")
    axes[0].set_title("Latency, one part at a time")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    axes[0].grid(axis="y", visible=False)

    piv = df.pivot(index="model", columns="batch_size", values="images_per_sec").loc[d1.index]
    w = 0.38
    for i, b in enumerate(piv.columns):
        bars = axes[1].barh(y + (i - 0.5) * w, piv[b], w, color=colors,
                            alpha=1.0 if i else 0.55)
        axes[1].bar_label(bars, fmt="%.0f", padding=2, fontsize=7.5, color=INK_2)
    axes[1].axvline(line_speed, color=PALETTE[7], ls="--", lw=1.2)
    _log_ticks(axes[1], [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Images per second (log scale)")
    axes[1].set_title(f"Throughput vs line speed ({line_speed} parts/s)")
    axes[1].annotate("upper bar = batch 1 (pale), lower bar = batch 32", (0.5, -0.2),
                     xycoords="axes fraction", ha="center", va="top", fontsize=8, color=INK_2)
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, tagged(cfg, "latency"), cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = apply_fast(load_config(), args.fast)
    set_seed(cfg["seed"])
    plot_style()
    device = get_device()
    lat = cfg["latency"]
    warmup, runs = lat["warmup_runs"], lat["timed_runs"]
    if cfg["is_fast"]:
        warmup, runs = 3, 20
    size = cfg["image"]["size"]
    line = lat["line_speed_parts_per_sec"]

    pre_s = time_preprocess(cfg)
    rows = []
    for name in MODEL_NAMES:
        path = get_path(cfg, "models_dir") / f"{tagged(cfg, name)}.pt"
        if not path.exists():
            print(f"  (skip {name}: no checkpoint)")
            continue
        model, _ = load_checkpoint(path, device)
        total, _ = count_params(model)
        for b in lat["batch_sizes"]:
            t = time_model(model, size, b, warmup, runs, device) / b * 1000  # ms per image
            ips = 1000 / np.median(t)
            rows.append({"model": name, "batch_size": b, "median_ms_per_image": np.median(t),
                         "p5_ms_per_image": np.percentile(t, 5),
                         "p95_ms_per_image": np.percentile(t, 95),
                         "end_to_end_ms": np.median(t) + 1000 * pre_s,
                         "images_per_sec": ips, "headroom_vs_line": ips / line,
                         "params": total, "checkpoint_mb": path.stat().st_size / 1e6})
            print(f"  {name:<20} batch {b:>2}: {np.median(t):7.2f} ms/img "
                  f"(p95 {np.percentile(t, 95):.2f}), {ips:7.1f} img/s, "
                  f"{ips / line:.0f}x line speed")
    df = pd.DataFrame(rows)
    metrics_dir = get_path(cfg, "metrics_dir")
    df.to_csv(metrics_dir / f"{tagged(cfg, 'latency')}.csv", index=False)
    save_json({"device": str(device), "cpu_threads": torch.get_num_threads(),
               "torch": torch.__version__, "platform": platform.platform(),
               "image_size": size, "warmup_runs": warmup, "timed_runs": runs,
               "preprocess_ms_median": 1000 * pre_s, "line_speed_parts_per_sec": line,
               "line_budget_ms_per_part": 1000 / line, "results": df.to_dict("records"),
               "note": "Timed on the machine above; other running processes inflate numbers."},
              metrics_dir / f"{tagged(cfg, 'latency')}.json")
    fig_latency(df, line, cfg)
    print(f"Preprocessing: {1000 * pre_s:.1f} ms/img median; line budget {1000 / line:.0f} ms/part")


if __name__ == "__main__":
    main()
