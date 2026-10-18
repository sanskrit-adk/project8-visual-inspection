"""Dataset download, indexing, statistics, splits, transforms and loaders.

Phase 1: download, index, per-image stats, near-duplicate check.
Phase 2: group-aware splits with defect subsampling, transforms, Dataset, loaders.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import ROOT, get_path

IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp"}


# ---------------------------------------------------------------------
# Download + index
# ---------------------------------------------------------------------
def download_casting(cfg: dict) -> Path:
    """Download the casting dataset with kagglehub and return the casting_data folder.

    KAGGLEHUB_CACHE is pointed at data/ so the images live inside the project
    (git-ignored) instead of a hidden user cache - every teammate gets the same
    relative paths, which keeps splits/*.csv portable.
    """
    os.environ.setdefault("KAGGLEHUB_CACHE", str(get_path(cfg, "data_dir") / "kagglehub"))
    import kagglehub  # imported after setting the cache dir on purpose

    root = Path(kagglehub.dataset_download(cfg["data"]["kaggle_casting"]))
    # The archive nests casting_data/casting_data/{train,test}; fall back to one level.
    for cand in (root / "casting_data" / "casting_data", root / "casting_data"):
        if (cand / "train").exists():
            return cand
    raise FileNotFoundError(f"casting_data/train not found under {root}")


def download_neu(cfg: dict) -> Path:
    """Download NEU-DET (steel surface defects, 6 types) and return its NEU-DET folder."""
    os.environ.setdefault("KAGGLEHUB_CACHE", str(get_path(cfg, "data_dir") / "kagglehub"))
    import kagglehub

    root = Path(kagglehub.dataset_download(cfg["data"]["kaggle_neu"]))
    for cand in (root / "NEU-DET", root):
        if (cand / "train" / "images").exists():
            return cand
    raise FileNotFoundError(f"NEU-DET/train/images not found under {root}")


def build_neu_index(neu_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """One row per NEU image: path, defect type, class id, original split, boxes.

    Boxes (from the Pascal-VOC XML files) are stored as JSON [[xmin,ymin,xmax,ymax],...]
    in pixel coordinates of the original 200x200 image; Phase 7 uses them to check
    whether Grad-CAM points at the annotated defect.
    """
    import json
    import xml.etree.ElementTree as ET

    classes = sorted(p.name for p in (neu_dir / "train" / "images").iterdir() if p.is_dir())
    rows = []
    for split in ("train", "validation"):
        for cid, cls in enumerate(classes):
            for p in sorted((neu_dir / split / "images" / cls).iterdir()):
                if p.suffix.lower() not in IMG_EXTS:
                    continue
                xml = neu_dir / split / "annotations" / f"{p.stem}.xml"
                boxes = []
                if xml.exists():
                    for obj in ET.parse(xml).getroot().iter("object"):
                        b = obj.find("bndbox")
                        boxes.append([int(float(b.find(k).text))
                                      for k in ("xmin", "ymin", "xmax", "ymax")])
                rows.append({"path": _rel(p), "filename": p.name, "class": cls, "label": cid,
                             "orig_split": split, "boxes": json.dumps(boxes)})
    return pd.DataFrame(rows), classes


def _rel(p: Path) -> str:
    """Store paths relative to the project root so CSVs work on any machine."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:  # data stored outside the project: keep absolute
        return p.resolve().as_posix()


def resolve(path: str) -> Path:
    """Inverse of _rel: turn an index path back into an absolute path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_index(casting_dir: Path, cfg: dict) -> pd.DataFrame:
    """One row per image: path, class, label (defect=1), original Kaggle split."""
    pos, neg = cfg["data"]["positive_class"], cfg["data"]["negative_class"]
    rows = []
    for split in ("train", "test"):
        for cls, label in ((neg, 0), (pos, 1)):
            for p in sorted((casting_dir / split / cls).iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    rows.append({"path": _rel(p), "filename": p.name, "class": cls,
                                 "label": label, "orig_split": split})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Per-image statistics (for EDA + the shortcut baseline)
# ---------------------------------------------------------------------
def radial_masks(h: int, w: int, part_frac: float, bg_frac: float):
    """Boolean masks for the central part and the surrounding background.

    Radii are fractions of the image width so the masks work at any resolution.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2) / w
    return r <= part_frac, r >= bg_frac


def image_stats(path: Path, cfg: dict) -> tuple[dict, np.ndarray]:
    """Cheap hand-crafted statistics + a small thumbnail for duplicate search.

    These are deliberately 'dumb' features: if a classifier on them alone can
    separate defects from good parts, then something about *how the photos were
    taken* (lighting, exposure, background) differs between classes - a
    shortcut a CNN could also learn instead of the defect itself.
    """
    import cv2  # local import: only needed here

    img = Image.open(path)
    w, h = img.size
    mode = img.mode
    g = np.asarray(img.convert("L"), dtype=np.float32)

    part, bg = radial_masks(h, w, cfg["data"]["part_radius_frac"],
                            cfg["data"]["background_radius_frac"])
    # Canny with fixed thresholds -> fraction of pixels on an edge. Cracks and
    # pits add edges, so this feature can carry *real* defect signal too.
    edges = cv2.Canny(g.astype(np.uint8), 50, 150) > 0

    stats = {
        "width": w, "height": h, "mode": mode,
        "mean": float(g.mean()),             # overall brightness (exposure / lighting)
        "std": float(g.std()),               # global contrast
        "edge_density": float(edges.mean()),
        "part_mean": float(g[part].mean()),
        "part_std": float(g[part].std()),
        "bg_mean": float(g[bg].mean()),      # background only: should carry NO defect info
        "bg_std": float(g[bg].std()),
    }

    s = cfg["data"]["dup_thumb_size"]
    t = np.asarray(Image.fromarray(g.astype(np.uint8)).resize((s, s), Image.BILINEAR),
                   dtype=np.float32).ravel()
    t -= t.mean()
    t /= np.linalg.norm(t) + 1e-8            # unit vector -> dot product = Pearson correlation
    return stats, t


# ---------------------------------------------------------------------
# Near-duplicate search
# ---------------------------------------------------------------------
def nearest_neighbours(query: np.ndarray, ref: np.ndarray, exclude_self: bool,
                       chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """For each query thumbnail, the best correlation with any reference thumbnail.

    Chunked so the full similarity matrix never sits in memory at once.
    """
    best = np.empty(len(query), dtype=np.float32)
    arg = np.empty(len(query), dtype=np.int64)
    for i in range(0, len(query), chunk):
        s = query[i:i + chunk] @ ref.T
        if exclude_self:
            idx = np.arange(i, min(i + chunk, len(query)))
            s[idx - i, idx] = -1.0
        arg[i:i + chunk] = s.argmax(1)
        best[i:i + chunk] = s.max(1)
    return best, arg


def duplicate_groups(thumbs: np.ndarray, threshold: float, chunk: int = 1024) -> np.ndarray:
    """Connected components of the 'correlation >= threshold' graph.

    Images in the same component are probably copies / augmentations of one
    physical part. Phase 2 can use these ids for a group-aware split so that
    copies of the same part never sit on both sides of train/test.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows, cols = [], []
    for i in range(0, len(thumbs), chunk):
        s = thumbs[i:i + chunk] @ thumbs.T
        r, c = np.nonzero(s >= threshold)
        rows.append(r + i)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(len(thumbs),) * 2)
    _, labels = connected_components(g, directed=False)
    return labels


# ---------------------------------------------------------------------
# Splits (Phase 2)
# ---------------------------------------------------------------------
SPLITS = ("train", "val", "test", "production")


def split_fractions(cfg: dict) -> dict[str, float]:
    """Target share of the data per split (production first, the rest scaled)."""
    d = cfg["data"]
    p = d["production_frac"]
    return {"production": p, **{k: v * (1 - p) for k, v in d["split"].items()}}


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Deal whole duplicate groups into train/val/test/production.

    Greedy balancing: groups are visited largest first (ties in random order) and
    each goes to the split that is furthest below its target share of OK images
    and of defect images, weighted by how many of each the group holds. Largest
    first matters because a few groups hold 300+ copies; placed late they would
    overshoot whichever split received them. Every copy of a part lands in
    exactly one split.
    """
    frac = split_fractions(cfg)
    names = list(frac)
    g = df.groupby("dup_group")["label"].agg(n_def="sum", n="size")
    g["n_ok"] = g["n"] - g["n_def"]
    rng = np.random.default_rng(cfg["seed"])
    g = g.assign(_r=rng.random(len(g))).sort_values(["n", "_r"], ascending=[False, True])

    tgt_ok = np.array([frac[s] * g["n_ok"].sum() for s in names])
    tgt_def = np.array([frac[s] * g["n_def"].sum() for s in names])
    cur_ok, cur_def = np.zeros(len(names)), np.zeros(len(names))
    choice = {}
    for gid, row in g.iterrows():
        need = row["n_ok"] * (1 - cur_ok / tgt_ok) + row["n_def"] * (1 - cur_def / tgt_def)
        s = int(np.argmax(need))
        choice[gid] = names[s]
        cur_ok[s] += row["n_ok"]
        cur_def[s] += row["n_def"]
    return df["dup_group"].map(choice).rename("split")


def subsample_defects(df: pd.DataFrame, ratio: float, seed: int) -> pd.DataFrame:
    """Keep all OK images and enough defects that they make up `ratio` of the split.

    Defects are taken round-robin across duplicate groups (one per group before a
    second from any group), so the few defects we keep cover as many distinct
    physical parts as possible instead of many copies of the same few.
    """
    ok, de = df[df["label"] == 0], df[df["label"] == 1]
    n_def = min(len(de), round(len(ok) * ratio / (1 - ratio)))
    rng = np.random.default_rng(seed)
    de = de.assign(_r=rng.random(len(de))).sort_values("_r")
    de = de.assign(_rank=de.groupby("dup_group").cumcount()).sort_values(["_rank", "_r"])
    keep = de.head(n_def).drop(columns=["_r", "_rank"])
    return pd.concat([ok, keep]).sort_index()


def fast_subset(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Stratified random subset for --fast runs (keeps the defect ratio)."""
    return df.groupby("label").sample(frac=frac, random_state=seed).sort_index()


def load_split(cfg: dict, name: str) -> pd.DataFrame:
    """Read splits/<name>.csv, shrunk to the fast subset when cfg['is_fast']."""
    df = pd.read_csv(get_path(cfg, "splits_dir") / f"{name}.csv")
    if cfg.get("is_fast"):
        df = fast_subset(df, cfg["fast"]["subset_fraction"], cfg["seed"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Transforms, Dataset, loaders (Phase 2)
# ---------------------------------------------------------------------
def build_transforms(cfg: dict, train: bool):
    """Train: geometric + photometric augmentation. Eval: resize + normalise only.

    Flips and rotations are label-preserving for a round casting seen from the
    front. Brightness/contrast jitter also blurs the background-brightness gap
    between classes found in Phase 1, making that shortcut less reliable.
    """
    from torchvision import transforms as T

    size = cfg["image"]["size"]
    norm = [T.ToTensor(), T.Normalize(cfg["image"]["mean"], cfg["image"]["std"])]
    if not train:
        return T.Compose([T.Resize((size, size)), *norm])
    a = cfg["augmentation"]
    aug = [T.Resize((size, size))]
    if a["hflip"]:
        aug.append(T.RandomHorizontalFlip())
    if a["vflip"]:
        aug.append(T.RandomVerticalFlip())
    aug += [
        T.RandomRotation(a["rotation_deg"], fill=(a["rotation_fill"],) * 3),
        T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        T.RandomApply([T.GaussianBlur(5, sigma=tuple(a["blur_sigma"]))], p=a["blur_prob"]),
    ]
    return T.Compose(aug + norm)


class CastingDataset:
    """Images listed in a split DataFrame -> (tensor, float label).

    Order follows the DataFrame, so with shuffle=False predictions line up with
    `dataset.df` rows (used for error galleries and per-image results).
    """

    def __init__(self, df: pd.DataFrame, transform, label_dtype=np.float32):
        # float32 labels for the binary BCE models; int64 class ids for NEU (cross-entropy).
        self.df = df.reset_index(drop=True)
        self.paths = [resolve(p) for p in self.df["path"]]
        self.labels = self.df["label"].to_numpy(dtype=label_dtype)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def pos_weight(df: pd.DataFrame) -> float:
    """n_negative / n_positive, for BCEWithLogitsLoss(pos_weight=...)."""
    n_pos = int(df["label"].sum())
    return (len(df) - n_pos) / max(n_pos, 1)


def make_loaders(cfg: dict, splits=("train", "val", "test"), train_df: pd.DataFrame | None = None):
    """DataLoaders for the requested splits.

    Train uses a WeightedRandomSampler (each batch ~50/50) when
    cfg.train.imbalance == "sampler"; otherwise it is shuffled normally and the
    loss should use pos_weight(train_df). `train_df` overrides splits/train.csv
    (used by the data-scarcity experiment).
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    t = cfg["train"]
    g = torch.Generator().manual_seed(cfg["seed"])
    loaders = {}
    for name in splits:
        df = train_df if (name == "train" and train_df is not None) else load_split(cfg, name)
        is_train = name == "train"
        ds = CastingDataset(df, build_transforms(cfg, train=is_train))
        sampler, shuffle = None, False
        if is_train and t["imbalance"] == "sampler":
            counts = np.bincount(df["label"], minlength=2)
            w = 1.0 / counts[df["label"].to_numpy()]
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                            num_samples=len(df), replacement=True, generator=g)
        elif is_train:
            shuffle = True
        loaders[name] = DataLoader(ds, batch_size=t["batch_size"], shuffle=shuffle,
                                   sampler=sampler, generator=g if shuffle else None,
                                   num_workers=t["num_workers"],
                                   persistent_workers=t["num_workers"] > 0,
                                   pin_memory=torch.cuda.is_available())
    return loaders
