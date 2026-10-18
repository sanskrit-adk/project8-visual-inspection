"""Phase 9: web demo - inspect a casting image.

Loads the deployed model and threshold from Phase 5 (full-run results if they
exist, otherwise the --fast ones), then for a chosen or uploaded image shows the
defect probability, the accept/reject decision, and a Grad-CAM heatmap. A sidebar
slider changes the cost ratio and re-derives the threshold from the validation
predictions, so the cost trade-off from Phase 5 can be explored live.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data import build_transforms, resolve  # noqa: E402
from src.evaluation import choose_threshold, smoothed_cost_curve  # noqa: E402
from src.explain import compute_cams, make_cam, overlay  # noqa: E402
from src.models import load_checkpoint  # noqa: E402
from src.utils import get_path, load_config  # noqa: E402

st.set_page_config(page_title="Casting Inspection", page_icon="🔍", layout="wide")


@st.cache_resource
def load_everything():
    """Pick full-run artefacts if present, else fast ones."""
    cfg = load_config()
    metrics = get_path(cfg, "metrics_dir")
    for suffix in ("", "_fast"):
        thr_path = metrics / f"threshold{suffix}.json"
        if thr_path.exists():
            break
    else:
        return None
    info = json.loads(thr_path.read_text())
    cfg["image"]["size"] = info["image_size"]
    model, _ = load_checkpoint(get_path(cfg, "models_dir") / f"{info['model']}{suffix}.pt", "cpu")
    val = pd.read_csv(metrics / f"predictions_val{suffix}.csv")
    return {"cfg": cfg, "info": info, "model": model, "cam": make_cam(model), "val": val,
            "suffix": suffix, "tf": build_transforms(cfg, train=False)}


@st.cache_data
def threshold_for(ratio: float, suffix: str) -> float:
    s = load_everything()
    v, cfg = s["val"], s["cfg"]
    curve = smoothed_cost_curve(v["label"].to_numpy(), v[s["info"]["model"]].to_numpy(),
                                np.linspace(0, 1, cfg["evaluation"]["threshold_steps"]),
                                ratio * cfg["evaluation"]["cost_false_reject"],
                                cfg["evaluation"]["cost_false_reject"], cfg["data"]["defect_ratio"])
    return choose_threshold(curve)


@st.cache_data
def sample_table(suffix: str) -> pd.DataFrame:
    cfg = load_config()
    frames = []
    for split in ("test", "production"):
        p = get_path(cfg, "splits_dir") / f"{split}.csv"
        if p.exists():
            frames.append(pd.read_csv(p).assign(split=split))
    return pd.concat(frames, ignore_index=True)


def inspect(img: Image.Image, s: dict):
    x = s["tf"](img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = float(torch.sigmoid(s["model"](x))[0])
    heat = compute_cams(s["cam"], x)[0]
    size = s["cfg"]["image"]["size"]
    gray = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32) / 255
    return p, overlay(gray, heat)


# ---------------------------------------------------------------------
s = load_everything()
st.title("Automated visual inspection - cast parts")
if s is None:
    st.error("No trained model found. Run the pipeline first, e.g. "
             "`python scripts/run_all.py --fast`.")
    st.stop()

info, cfg = s["info"], s["cfg"]
with st.sidebar:
    st.header("Decision settings")
    if s["suffix"]:
        st.warning("Using **--fast** results (128px smoke run). Run the full pipeline for "
                   "report-quality numbers.")
    ratio = st.select_slider("Cost of a shipped defect : cost of a false reject",
                             options=[1, 2, 5, 10, 20, 50, 100, 200],
                             value=int(info["cost_escaped_defect"] / info["cost_false_reject"]))
    thr = threshold_for(float(ratio), s["suffix"])
    st.metric("Threshold", f"{thr:.3f}",
              help="Minimises expected cost on the validation split (smoothed cost curve).")
    st.caption(f"Model **{info['model']}** · line prevalence {info['line_prevalence']:.0%} · "
               f"Phase 5 threshold at {info['cost_escaped_defect']}:{info['cost_false_reject']} "
               f"= {info['threshold']:.3f}")
    t = info["test"]
    st.caption(f"Test at the Phase 5 threshold: recall {t['recall']:.2f}, "
               f"false-reject rate {t['false_reject_rate']:.1%}, "
               f"cost {t['cost_per_1000']:.0f} per 1000 parts.")

tab_pick, tab_upload = st.tabs(["Pick a held-out image", "Upload an image"])
with tab_pick:
    table = sample_table(s["suffix"])
    c1, c2 = st.columns(2)
    split = c1.radio("Split", sorted(table["split"].unique()), horizontal=True)
    kind = c2.radio("True class", ["Defect", "OK"], horizontal=True)
    pool = table[(table["split"] == split) & (table["label"] == (kind == "Defect"))]
    name = st.selectbox("Image", pool["filename"].tolist())
    row = pool[pool["filename"] == name].iloc[0]
    image, truth = Image.open(resolve(row["path"])), kind
with tab_upload:
    up = st.file_uploader("Casting photo (top view, like the dataset)", type=["jpg", "jpeg", "png"])
    if up is not None:
        image, truth = Image.open(up), None

p, cam_img = inspect(image, s)
defect = p >= thr
c1, c2, c3 = st.columns([1, 1, 1])
c1.image(image.convert("RGB"), caption="Input", width="stretch")
c2.image(cam_img, caption="Grad-CAM: evidence for 'defect'", width="stretch")
with c3:
    st.metric("Defect probability", f"{p:.3f}")
    if defect:
        st.error(f"REJECT - p {p:.3f} ≥ threshold {thr:.3f}")
    else:
        st.success(f"ACCEPT - p {p:.3f} < threshold {thr:.3f}")
    if truth is not None:
        correct = defect == (truth == "Defect")
        st.write(f"True class: **{truth}** - decision is "
                 f"{'✅ correct' if correct else '❌ wrong'}")
    st.caption("Probabilities are not calibrated (the model trained on ~50% defects), so "
               "compare them with the cost-based threshold, not with 0.5.")
