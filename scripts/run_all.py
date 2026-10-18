"""Run the whole pipeline, phase by phase, stopping at the first failure.

    python scripts/run_all.py --fast            # CPU smoke run, ~15-20 min
    python scripts/run_all.py                   # full run (GPU recommended)
    python scripts/run_all.py --from 4          # resume from Phase 4 (e.g. after training)
    python scripts/run_all.py --skip-scarcity   # skip the slow data-scarcity retraining
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (phase number, script, takes --fast)
PHASES = [
    (0, "00_check_setup.py", False),
    (1, "01_eda.py", False),
    (2, "02_splits.py", False),
    (3, "03_train_models.py", True),
    (4, "04_evaluate.py", True),
    (4.5, "04b_neu_per_type.py", True),
    (5, "05_threshold.py", True),
    (6, "06_latency.py", True),
    (7, "07_gradcam.py", True),
    (8, "08_robustness.py", True),
    (10, "10_make_report_notes.py", False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--from", dest="start", type=float, default=0)
    ap.add_argument("--skip-scarcity", action="store_true")
    args = ap.parse_args()

    t_all = time.perf_counter()
    for num, script, fastable in PHASES:
        if num < args.start:
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / script)]
        if args.fast and fastable:
            cmd.append("--fast")
        if script == "03_train_models.py" and args.skip_scarcity:
            cmd.append("--skip-scarcity")
        print(f"\n{'=' * 70}\nPhase {num:g}: {' '.join(cmd[1:])}\n{'=' * 70}", flush=True)
        t0 = time.perf_counter()
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit(f"\nPhase {num:g} ({script}) failed - fix it, then resume with --from {num:g}")
        print(f"[phase {num:g} done in {time.perf_counter() - t0:.0f}s]", flush=True)
    print(f"\nAll phases done in {(time.perf_counter() - t_all) / 60:.1f} min. "
          f"Demo: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
