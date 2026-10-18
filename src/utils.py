"""Shared helpers: config loading, seeding, device selection, paths, saving.

Every script imports from here so that seeds, devices and paths are handled
the same way everywhere. That consistency is what makes results reproducible
across team members' machines.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# Project root = the folder that contains config.yaml (one level above src/).
# Resolving paths from here means scripts work no matter which directory
# they are launched from.
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml as a plain dict."""
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_fast(cfg: dict, fast: bool) -> dict:
    """Override settings with the `fast:` block when --fast is passed.

    We overwrite values in place rather than branching in every script, so the
    rest of the code never needs to know whether it is in fast mode.
    """
    cfg["is_fast"] = bool(fast)
    if fast:
        f = cfg["fast"]
        cfg["image"]["size"] = f["image_size"]
        cfg["train"]["epochs"] = f["epochs"]
        cfg["train"]["patience"] = f["patience"]
        cfg["train"]["batch_size"] = f["batch_size"]
        cfg["train"]["num_workers"] = f["num_workers"]
    return cfg


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU + GPU).

    cudnn.deterministic trades a little GPU speed for repeatable runs, which
    matters more here than raw speed: reported numbers must be reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the fastest available device: CUDA -> Apple MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_path(cfg: dict, key: str) -> Path:
    """Absolute path for a key under `paths:` in the config; created if missing."""
    p = ROOT / cfg["paths"][key]
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path) -> None:
    """Save metrics as pretty JSON. NumPy types are converted so json can write them."""
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def save_fig(fig, name: str, cfg: dict) -> Path:
    """Save a matplotlib figure as PNG (dpi 150) into results/figures/."""
    out = get_path(cfg, "figures_dir") / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return out
