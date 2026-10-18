"""Image perturbations for the robustness and shortcut tests (Phase 8).

Each perturbation is a function PIL.Image -> PIL.Image applied BEFORE the usual
eval transform, so the model sees exactly what a changed camera/lighting setup
would produce.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from src.data import radial_masks


def brightness(f):
    return lambda im: ImageEnhance.Brightness(im).enhance(f)


def contrast(f):
    return lambda im: ImageEnhance.Contrast(im).enhance(f)


def blur(sigma):
    return lambda im: im.filter(ImageFilter.GaussianBlur(radius=sigma))


def rotation(deg, fill):
    return lambda im: im.rotate(deg, resample=Image.BILINEAR, fillcolor=(fill,) * 3)


def _masks(im, cfg):
    w, h = im.size
    return radial_masks(h, w, cfg["data"]["part_radius_frac"], cfg["data"]["background_radius_frac"])


def part_only(cfg, fill):
    """Everything outside the part radius replaced by one grey: no background cues left."""
    def f(im):
        a = np.asarray(im).copy()
        part, _ = _masks(im, cfg)
        a[~part] = fill
        return Image.fromarray(a)
    return f


def background_only(cfg, fill):
    """Everything inside the background radius (the whole casting) replaced by one grey.

    A model that still separates defects from OK parts here is reading the
    background - it cannot see the part at all.
    """
    def f(im):
        a = np.asarray(im).copy()
        _, bg = _masks(im, cfg)
        a[~bg] = fill
        return Image.fromarray(a)
    return f


def background_equalized(cfg, target):
    """Shift background pixels so every image's background has the same mean brightness.

    Keeps the background texture but removes the brightness difference between
    classes found in Phase 1. A model relying on that cue should drop here.
    """
    def f(im):
        a = np.asarray(im).astype(np.float32)
        part, _ = _masks(im, cfg)
        g = a.mean(2)
        a[~part] += target - g[~part].mean()
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return f


class PerturbedDataset:
    """Like CastingDataset, with `perturb` applied to the PIL image first."""

    def __init__(self, base, perturb):
        self.base, self.perturb = base, perturb

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img = Image.open(self.base.paths[i]).convert("RGB")
        return self.base.transform(self.perturb(img)), self.base.labels[i]
