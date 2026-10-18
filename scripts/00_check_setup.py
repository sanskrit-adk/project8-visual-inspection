"""Phase 0: confirm the environment is ready.

Prints the detected device and library versions, checks the config loads,
and writes them to results/metrics/environment.json so the report can state
exactly what software produced the results.

Run:  python scripts/00_check_setup.py
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.utils import get_device, get_path, load_config, save_json, set_seed  # noqa: E402

PACKAGES = ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
            "pyyaml", "pillow", "kagglehub", "grad-cam", "opencv-python-headless",
            "streamlit", "tqdm"]


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()

    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "NOT INSTALLED"

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_threads": torch.get_num_threads(),
        "seed": cfg["seed"],
        "packages": versions,
    }

    print(f"Python      : {info['python']}  ({info['platform']})")
    print(f"Device      : {info['device']}  (GPU: {info['gpu_name']})")
    print(f"CPU threads : {info['cpu_threads']}")
    print(f"Seed        : {info['seed']}")
    print("Packages:")
    for k, v in versions.items():
        print(f"  {k:<24}{v}")

    # Make sure every output folder exists so later phases can just write.
    for key in cfg["paths"]:
        get_path(cfg, key)

    # Tiny smoke test: a forward pass on the chosen device.
    x = torch.randn(2, 3, 32, 32, device=device)
    y = torch.nn.Conv2d(3, 4, 3).to(device)(x)
    print(f"Smoke test  : conv forward OK on {device}, output shape {tuple(y.shape)}")

    missing = [k for k, v in versions.items() if v == "NOT INSTALLED"]
    save_json(info, get_path(cfg, "metrics_dir") / "environment.json")
    if missing:
        print(f"\nWARNING: missing packages: {missing}")
        sys.exit(1)
    print("\nSetup OK.")


if __name__ == "__main__":
    main()
