"""Grad-CAM helpers shared by Phase 7 and the demo app."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models import gradcam_target_layer


class _TwoDimOutput(nn.Module):
    """pytorch-grad-cam expects (N, classes) outputs; binary models return (N,)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.unsqueeze(1) if out.dim() == 1 else out


def make_cam(model: nn.Module):
    """GradCAM on the model's last conv block. Model must be in eval mode."""
    from pytorch_grad_cam import GradCAM

    model.eval()
    return GradCAM(model=_TwoDimOutput(model), target_layers=[gradcam_target_layer(model)])


def compute_cams(cam, x: torch.Tensor, class_idx: int | list[int] = 0, batch: int = 16) -> np.ndarray:
    """(N, H, W) heatmaps in [0, 1] (each min-max scaled) for the given class.

    For the binary models class 0 is the single 'defect' logit, so the map always
    shows evidence FOR a defect - also on OK parts and false alarms.
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    idx = [class_idx] * len(x) if isinstance(class_idx, int) else list(class_idx)
    out = []
    for i in range(0, len(x), batch):
        targets = [ClassifierOutputTarget(c) for c in idx[i:i + batch]]
        out.append(cam(input_tensor=x[i:i + batch], targets=targets))
    return np.concatenate(out)


def overlay(gray_or_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap onto an image (float [0,1], HxW or HxWx3) -> uint8 RGB."""
    import matplotlib

    img = gray_or_rgb if gray_or_rgb.ndim == 3 else np.stack([gray_or_rgb] * 3, -1)
    colored = matplotlib.colormaps["inferno"](heat)[..., :3]
    return (255 * np.clip((1 - alpha) * img + alpha * colored, 0, 1)).astype(np.uint8)


def energy_fraction(cams: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Share of each heatmap's total mass that falls inside `mask` (HxW bool)."""
    tot = cams.reshape(len(cams), -1).sum(1) + 1e-8
    return cams[:, mask].sum(1) / tot


def peak_in(cams: np.ndarray, masks: np.ndarray | list) -> np.ndarray:
    """Whether each heatmap's maximum lies inside its mask (one mask, or one per image)."""
    hits = []
    for i, c in enumerate(cams):
        m = masks if isinstance(masks, np.ndarray) and masks.ndim == 2 else masks[i]
        r, col = np.unravel_index(np.argmax(c), c.shape)
        hits.append(bool(m[r, col]))
    return np.array(hits)
