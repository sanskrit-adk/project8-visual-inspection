"""Prediction and metrics shared by training (Phase 3) and evaluation (Phase 4+).

With 10% defects, accuracy is misleading (always saying "OK" scores 90%), so the
headline metric is average precision (area under the precision-recall curve),
plus recall at the fixed precision levels a line manager would accept.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    """Defect probabilities, labels and mean (unweighted) BCE loss over a loader."""
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += bce(logits, y).item()
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
    y = np.concatenate(labels).astype(int)
    return np.concatenate(probs), y, loss_sum / len(y)


def recall_at_precision(y: np.ndarray, p: np.ndarray, min_precision: float) -> tuple[float, float]:
    """Best recall achievable while precision >= min_precision, and the threshold for it.

    Returns (0.0, nan) when no threshold reaches that precision.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    # precision_recall_curve returns one more (prec, rec) point than thresholds;
    # the last point (recall 0) has no threshold, so drop it.
    ok = prec[:-1] >= min_precision
    if not ok.any():
        return 0.0, float("nan")
    i = np.argmax(np.where(ok, rec[:-1], -1))
    return float(rec[i]), float(thr[i])


def binary_metrics(y: np.ndarray, p: np.ndarray, cfg: dict) -> dict:
    out = {"average_precision": float(average_precision_score(y, p)),
           "roc_auc": float(roc_auc_score(y, p)),
           "prevalence": float(y.mean())}
    for mp in cfg["evaluation"]["fixed_precisions"]:
        r, t = recall_at_precision(y, p, mp)
        out[f"recall_at_p{int(mp * 100)}"] = r
        out[f"threshold_at_p{int(mp * 100)}"] = t
    return out


# ---------------------------------------------------------------------
# Uncertainty (Phase 4)
# ---------------------------------------------------------------------
def group_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], groups: np.ndarray,
                    metric, n_boot: int = 2000, seed: int = 42) -> dict[str, np.ndarray]:
    """Bootstrap a metric for several score vectors on the SAME resamples.

    Whole duplicate groups are resampled, not single images: copies of one part
    are not independent evidence, so image-level resampling would give
    intervals that are too narrow. Using shared resamples for every model makes
    paired differences (model A - model B) meaningful.
    Resamples with no defect or no OK image are skipped (metric undefined).
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    out = {k: [] for k in scores}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        for k, s in scores.items():
            out[k].append(metric(yb, s[idx]))
    return {k: np.asarray(v) for k, v in out.items()}


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(samples, a)), float(np.quantile(samples, 1 - a))


def confusion_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Confusion counts and rates when 'defect' means p >= thr."""
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "false_reject_rate": fp / (fp + tn) if fp + tn else float("nan")}


# ---------------------------------------------------------------------
# Cost-based thresholds (Phase 5)
# ---------------------------------------------------------------------
def cost_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, c_fn: float, c_fp: float,
               prevalence: float) -> dict[str, np.ndarray]:
    """Expected cost per inspected part at each threshold ('defect' = p >= t).

    cost(t) = c_fn * prevalence * miss_rate(t) + c_fp * (1 - prevalence) * false_reject_rate(t)

    Rates are measured on the evaluation data but weighted by the LINE prevalence,
    so the same curve answers "what would this cost on a line with X% defects"
    even if the evaluation split has a different defect share.
    """
    pred = p[None, :] >= thresholds[:, None]
    pos, neg = y == 1, y == 0
    recall = pred[:, pos].mean(1)
    frr = pred[:, neg].mean(1)
    cost = c_fn * prevalence * (1 - recall) + c_fp * (1 - prevalence) * frr
    return {"threshold": thresholds, "cost": cost, "recall": recall, "false_reject_rate": frr}


def choose_threshold(curve: dict) -> float:
    """Threshold with minimum expected cost; middle of the flat minimum when tied.

    With a few hundred images the cost curve is a step function, so many
    thresholds tie. The middle of the first run of tied thresholds sits as far as
    possible from the steps on either side, so it is least sensitive to small
    shifts in the model's scores.
    """
    c, t = curve["cost"], curve["threshold"]
    is_min = np.isclose(c, c.min(), rtol=0, atol=1e-12)
    start = int(np.argmax(is_min))
    end = start
    while end + 1 < len(c) and is_min[end + 1]:
        end += 1
    return float((t[start] + t[end]) / 2)


def expected_cost(y, p, thr, c_fn, c_fp, prevalence) -> float:
    return float(cost_curve(y, p, np.array([thr]), c_fn, c_fp, prevalence)["cost"][0])


def prior_corrected(p: np.ndarray, train_prior: float, line_prior: float) -> np.ndarray:
    """Re-weight probabilities learned at `train_prior` defects to a line with `line_prior`.

    Bayes' rule on the odds: posterior odds scale with prior odds. The sampler
    showed the model ~50% defects, so its raw probabilities are too high for a
    10%-defect line.
    """
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p) * (line_prior / (1 - line_prior)) / (train_prior / (1 - train_prior))
    return odds / (1 + odds)


def smoothed_cost_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray, c_fn: float,
                        c_fp: float, prevalence: float) -> dict[str, np.ndarray]:
    """Cost curve from kernel-density estimates of each class's scores (logit scale).

    The empirical curve only knows the scores it has seen: with ~50 val defects
    it is a step function, and its minimum often sits just below the lowest-
    scored defect - so the next, slightly weaker defect is shipped. Smoothing
    gives every class a tail beyond its observed scores, so a 50:1 cost ratio
    pulls the threshold away from the defects into the good parts. Logit scale
    because model scores pile up near 0 and 1.
    """
    from scipy.stats import gaussian_kde

    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    zt = np.log(np.clip(thresholds, eps, 1 - eps) / (1 - np.clip(thresholds, eps, 1 - eps)))
    out = {}
    for lab, key in ((1, "recall"), (0, "false_reject_rate")):
        kde = gaussian_kde(z[y == lab])
        # P(score >= t) = 1 - CDF(t); integrate_box_1d gives the CDF.
        out[key] = np.array([kde.integrate_box_1d(t, np.inf) for t in zt])
    cost = c_fn * prevalence * (1 - out["recall"]) + c_fp * (1 - prevalence) * out["false_reject_rate"]
    return {"threshold": thresholds, "cost": cost, **out}
