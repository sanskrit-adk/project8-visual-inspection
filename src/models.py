"""Model definitions, parameter groups, checkpoint save/load.

Four models answer two questions:
  * Does ImageNet pretraining help on grey castings?   scratch_cnn vs resnet18_*
  * Is fine-tuning worth it over a frozen backbone?     resnet18_frozen vs resnet18_finetune
  mobilenet_v3_small is the cheap option for the latency trade-off (Phase 6).

Every model outputs one logit per image (defect = positive class).
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

MODEL_NAMES = ["scratch_cnn", "resnet18_frozen", "resnet18_finetune", "mobilenet_v3_small"]


class ScratchCNN(nn.Module):
    """Four conv blocks + global average pooling: a small baseline with no pretraining.

    Global pooling makes it work at any input size (128px fast mode, 224px full).
    """

    def __init__(self, widths=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks, c_in = [], 3
        for c in widths:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class TransferNet(nn.Module):
    """A torchvision ImageNet backbone with a new 1-logit head.

    freeze=True trains only the head. The frozen backbone is also kept in eval
    mode so its BatchNorm statistics stay the ImageNet ones instead of drifting
    towards our small dataset.
    """

    def __init__(self, arch: str, freeze: bool, pretrained: bool = True, num_outputs: int = 1):
        super().__init__()
        from torchvision import models as tvm

        self.arch, self.freeze = arch, freeze
        if arch == "resnet18":
            w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = tvm.resnet18(weights=w)
            self.net.fc = nn.Linear(self.net.fc.in_features, num_outputs)
            self.head_module = self.net.fc
        elif arch == "mobilenet_v3_small":
            w = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = tvm.mobilenet_v3_small(weights=w)
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, num_outputs)
            self.head_module = self.net.classifier
        else:
            raise ValueError(arch)

        if freeze:
            head_ids = {id(p) for p in self.head_module.parameters()}
            for p in self.net.parameters():
                p.requires_grad = id(p) in head_ids

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and mode:
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        out = self.net(x)
        # Binary models return one logit per image; multi-class (NEU) keep all columns.
        return out.squeeze(1) if out.shape[1] == 1 else out


def build_model(name: str, pretrained: bool = True, num_outputs: int = 1) -> nn.Module:
    if name == "scratch_cnn":
        return ScratchCNN()
    if name == "resnet18_frozen":
        return TransferNet("resnet18", freeze=True, pretrained=pretrained, num_outputs=num_outputs)
    if name == "resnet18_finetune":
        return TransferNet("resnet18", freeze=False, pretrained=pretrained, num_outputs=num_outputs)
    if name == "mobilenet_v3_small":
        return TransferNet("mobilenet_v3_small", freeze=False, pretrained=pretrained,
                           num_outputs=num_outputs)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """Optimizer groups: new head at lr_head, pretrained backbone at the lower lr_backbone.

    A from-scratch model has nothing to protect, so all its weights use lr_scratch.
    """
    t = cfg["train"]
    if not isinstance(model, TransferNet):
        return [{"params": model.parameters(), "lr": t["lr_scratch"]}]
    head = [p for p in model.head_module.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head}
    back = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": t["lr_head"]}]
    if back:
        groups.append({"params": back, "lr": t["lr_backbone"]})
    return groups


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block - the layer Grad-CAM reads (Phase 7)."""
    if isinstance(model, ScratchCNN):
        return model.features
    if model.arch == "resnet18":
        return model.net.layer4[-1]
    return model.net.features[-1]


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------
def save_checkpoint(model: nn.Module, name: str, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": name, "state_dict": model.state_dict(), **meta}, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the model from its name and load weights (no ImageNet download needed)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], pretrained=False,
                        num_outputs=ckpt.get("num_outputs", 1))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
