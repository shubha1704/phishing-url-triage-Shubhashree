import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

import phish_core as core  # the saved pipelines import their feature step from here

ROOT = Path(__file__).parent
st.set_page_config(page_title="Phish Triage Desk", page_icon="🎣", layout="wide")


@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    models = {n: joblib.load(ROOT / "model" / f) for n, f in core.MODEL_FILES.items()}
    return models, json.loads((ROOT / "model" / "meta.json").read_text())


@st.cache_data(show_spinner="Scoring with all five models...")
def score_all(df: pd.DataFrame):
    """Inference only - nothing is retrained on the uploaded file."""
    models, _ = load_models()
    return {n: m.predict_proba(df)[:, 1] for n, m in models.items()}


def check(df, meta):
    if core.RAW_TARGET not in df.columns:
        return f"Add the target column '{core.RAW_TARGET}' (1 = legitimate, 0 = phishing)."
    if not set(df[core.RAW_TARGET].dropna().unique()) <= {0, 1}:
        return f"'{core.RAW_TARGET}' must contain only 0 (phishing) and 1 (legitimate)."
    missing = [c for c in meta["raw_columns"] if c not in df.columns and c not in core.TEXT_COLS]
    return f"Missing feature columns: {', '.join(missing[:6])}..." if missing else None


def cm_figure(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Reds")
    ax.set_xticks([0, 1], core.CLASS_NAMES)
    ax.set_yticks([0, 1], core.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


try:
    models, meta = load_models()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load models ({e}). Run `python train.py` and commit the `model/` folder.")
    st.stop()

if meta["versions"]["scikit-learn"] != sklearn.__version__:
    st.warning(f"Models were trained on scikit-learn {meta['versions']['scikit-learn']}, "
               f"this server has {sklearn.__version__}. Pin the version in requirements.txt.")

st.title("🎣 Phish Triage Desk")
st.caption("Five saved classifiers score web pages as **Phishing** or **Legitimate**. "
           "Pick a model, tune how aggressive the filter is, and see what it costs a security team.")

with st.sidebar:
    st.header("1 · Test data")
    up = st.file_uploader("Upload test CSV (features + 'label')", type="csv")
    bundled = ROOT / "test_data.csv"
    if up is None and bundled.exists():
        st.info("Using the bundled hold-out file.")
    if bundled.exists():
        st.download_button("Download bundled test_data.csv", bundled.read_bytes(), "test_data.csv")
    st.header("2 · Model")
    choice = st.selectbox("Classifier", list(models))
    st.header("3 · Analyst settings")
    thr = st.slider("Block a page when P(phishing) ≥", 0.05, 0.95, 0.50, 0.05)
    cost_fn = st.number_input("Cost of a missed phishing page", 1, 10000, 100)
    cost_fp = st.number_input("Cost of blocking a safe page", 1, 10000, 5)

data = pd.read_csv(up) if up is not None else (pd.read_csv(bundled) if bundled.exists() else None)
if data is None:
    st.stop()
data.columns = [c.strip() for c in data.columns]
problem = check(data, meta)
if problem:
    st.error(problem)
    st.stop()

y = core.to_binary(data[core.RAW_TARGET])
probas = score_all(data.drop(columns=[core.RAW_TARGET]))
p = probas[choice]

t_eval, t_all, t_flag = st.tabs(["📊 Model evaluation", "🏁 All five models", "🚨 Flagged pages & cost"])

with t_eval:
    st.subheader(f"{choice} on {len(data):,} pages (default 0.5 threshold)")
    m = core.evaluate(y, (p >= 0.5).astype(int), p)
    for col, k in zip(st.columns(6), core.METRIC_ORDER):
        col.metric(k, "n/a" if np.isnan(m[k]) else f"{m[k]:.4f}")
    a, b = st.columns([1, 1.2])
    a.pyplot(cm_figure(y, (p >= 0.5).astype(int)))
    rep = classification_report(y, (p >= 0.5).astype(int), labels=[0, 1],
                                target_names=core.CLASS_NAMES, output_dict=True, zero_division=0)
    b.dataframe(pd.DataFrame(rep).T.round(4))
    b.caption("Precision / Recall / F1 are reported for the Phishing class (the threat).")

with t_all:
    table = pd.DataFrame({n: core.evaluate(y, (pp >= 0.5).astype(int), pp) for n, pp in probas.items()}).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8cf"))
    st.bar_chart(table[["Recall", "Precision", "MCC"]])

with t_flag:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    total_cost = fn * cost_fn + fp * cost_fp
    c = st.columns(4)
    c[0].metric("Pages blocked", f"{pred.sum():,}")
    c[1].metric("Phishing missed", f"{fn:,}")
    c[2].metric("Safe pages blocked", f"{fp:,}")
    c[3].metric("Total cost", f"{total_cost:,}")
    grid = np.arange(0.05, 0.96, 0.05)
    costs = [((p >= g).astype(int) < y).sum() * cost_fn + ((p >= g).astype(int) > y).sum() * cost_fp
             for g in grid]
    st.line_chart(pd.DataFrame({"total cost": costs}, index=np.round(grid, 2)))
    st.caption(f"Cheapest threshold for these costs: **{grid[int(np.argmin(costs))]:.2f}**")
    show = pd.DataFrame({"P(phishing)": p.round(4), "blocked": pred,
                         "actually_phishing": y})
    if "URL" in data.columns:
        show.insert(0, "URL", data["URL"].values)
    flagged = show[show["blocked"] == 1].sort_values("P(phishing)", ascending=False)
    st.dataframe(flagged.head(300), hide_index=True)
    st.download_button("Download full triage report", show.to_csv(index=False).encode(),
                       "triage_report.csv")
