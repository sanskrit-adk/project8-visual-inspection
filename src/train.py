"""Training loop with early stopping on validation average precision."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from src.data import pos_weight
from src.evaluation import binary_metrics, predict
from src.models import param_groups, save_checkpoint


def make_criterion(cfg: dict, train_df, device) -> nn.Module:
    """Plain BCE when the sampler already balances batches; pos_weight BCE otherwise.

    Using both at once would correct the imbalance twice and push the model to
    over-predict defects.
    """
    if cfg["train"]["imbalance"] == "weighted_loss":
        pw = torch.tensor(pos_weight(train_df), device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.BCEWithLogitsLoss()


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


def fit(model, name: str, loaders: dict, cfg: dict, device, ckpt_path=None, verbose=True) -> dict:
    """Train `model`, keep the weights from the epoch with the best validation AP.

    Early stopping uses AP rather than validation loss because AP is the metric
    we report and select on; loss can keep falling while ranking of the rare
    defects stops improving.
    """
    t = cfg["train"]
    model.to(device)
    train_df = loaders["train"].dataset.df
    criterion = make_criterion(cfg, train_df, device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])

    history, best_ap, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        e0 = time.perf_counter()
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        scheduler.step()
        p, y, va_loss = predict(model, loaders["val"], device)
        m = binary_metrics(y, p, cfg)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_ap": m["average_precision"], "val_auc": m["roc_auc"],
                        "epoch_sec": time.perf_counter() - e0})
        improved = m["average_precision"] > best_ap
        if improved:
            best_ap, best_epoch, bad = m["average_precision"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"  [{name}] epoch {epoch:>2}  train {tr_loss:.4f}  val {va_loss:.4f}  "
                  f"AP {m['average_precision']:.4f}  AUC {m['roc_auc']:.4f}"
                  f"{'  *' if improved else ''}  ({history[-1]['epoch_sec']:.0f}s)")
        if bad >= t["patience"]:
            if verbose:
                print(f"  [{name}] early stop: no AP gain for {bad} epochs")
            break

    model.load_state_dict(best_state)
    p, y, va_loss = predict(model, loaders["val"], device)
    val_metrics = binary_metrics(y, p, cfg)
    result = {"best_epoch": best_epoch, "epochs_run": len(history),
              "train_sec": time.perf_counter() - start, "val": val_metrics,
              "val_loss": va_loss, "history": history,
              "val_probs": p.astype(np.float32), "val_labels": y}
    if ckpt_path is not None:
        save_checkpoint(model, name, ckpt_path, image_size=cfg["image"]["size"],
                        best_epoch=best_epoch, val_ap=val_metrics["average_precision"],
                        imbalance=t["imbalance"], is_fast=cfg.get("is_fast", False))
    return result


# ---------------------------------------------------------------------
# Multi-class (NEU defect types, Phase 4b)
# ---------------------------------------------------------------------
@torch.no_grad()
def predict_multiclass(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities (n, k) and integer labels."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def fit_multiclass(model, name: str, loaders: dict, cfg: dict, device, verbose=True) -> dict:
    """Cross-entropy training with early stopping on validation macro-F1."""
    from sklearn.metrics import f1_score

    t = cfg["train"]
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(param_groups(model, cfg), weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    history, best, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    start = time.perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        tr_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        scheduler.step()
        p, y = predict_multiclass(model, loaders["val"], device)
        f1 = f1_score(y, p.argmax(1), average="macro")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_macro_f1": f1})
        improved = f1 > best
        if improved:
            best, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"  [{name}] epoch {epoch:>2}  train {tr_loss:.4f}  val macro-F1 {f1:.4f}"
                  f"{'  *' if improved else ''}")
        if bad >= t["patience"]:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_run": len(history), "val_macro_f1": best,
            "train_sec": time.perf_counter() - start, "history": history}
