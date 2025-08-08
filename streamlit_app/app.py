# streamlit_app/app.py
# ------------------------------------------------------
# Fraud Detector – Better Minimal Demo
# - Loads pipeline + best_threshold from reports/metrics.json
# - Single-sample form + batch CSV upload
# - Adjustable threshold, metrics if 'Class' present
# - Saves predictions to reports/predictions_streamlit.csv
# ------------------------------------------------------

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

MODEL_PATH = Path("models/fraud_pipeline.joblib")
METRICS_PATH = Path("reports/metrics.json")
PRED_OUT = Path("reports/predictions_streamlit.csv")

# ---------- utils ----------

@st.cache_resource
def load_artifacts():
    assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    best_thr = 0.5
    if METRICS_PATH.exists():
        try:
            m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            best_thr = float(m.get("extra", {}).get("best_threshold", 0.5))
        except Exception:
            pass
    feature_cols = infer_feature_names(model)
    return model, best_thr, feature_cols

def infer_feature_names(model) -> List[str]:
    # Try to get feature names from pipeline; fallback to common creditcard schema
    names = None
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
    elif hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "get_feature_names_out"):
                try:
                    names = list(step.get_feature_names_out())
                    break
                except Exception:
                    pass
            if hasattr(step, "feature_names_in_"):
                names = list(step.feature_names_in_)
                break
    if names is None or len(names) == 0:
        # Fallback for Kaggle creditcard dataset
        names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return names

def ensure_column_order(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Add missing columns with 0.0 and drop extras not used by model
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_cols].copy()

def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        # Scale decision_function to [0,1] as a fallback
        s = model.decision_function(X)
        s = (s - s.min()) / max(1e-12, (s.max() - s.min()))
        return s
    raise RuntimeError("Model does not support predict_proba/decision_function")

def plot_hist(probs: np.ndarray, thr: float):
    fig, ax = plt.subplots()
    ax.hist(probs, bins=50)
    ax.axvline(thr, linestyle="--")
    ax.set_title("Score distribution")
    ax.set_xlabel("Fraud probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

def show_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2), yticks=np.arange(2),
        xticklabels=[0,1], yticklabels=[0,1],
        xlabel="Predicted", ylabel="True", title="Confusion Matrix"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

# ---------- UI ----------

st.set_page_config(page_title="اختلال‌یاب – Demo", layout="wide")
st.title("اختلال‌یاب — Fraud Detection (Demo)")

model, best_thr, feature_cols = load_artifacts()

with st.sidebar:
    st.header("Settings")
    st.write(f"Loaded model: `{MODEL_PATH}`")
    use_thr = st.slider("Threshold", min_value=0.05, max_value=0.95, step=0.01, value=float(best_thr))
    st.caption("Tip: این مقدار از metrics.json خوانده شده و قابل تغییر است.")

tabs = st.tabs(["Single Input", "Batch CSV"])

# ----- Single Input -----
with tabs[0]:
    st.subheader("Single Prediction")
    st.caption("مقادیر را وارد کن (اگر ستونی ندادی، مقدارش 0 در نظر گرفته می‌شود).")
    cols = st.columns(4)
    inputs = {}
    # چند فیلد مهم را جلو چشم می‌گذاریم؛ بقیه با JSON هم می‌شود
    quick_fields = ["Time", "Amount", "V1", "V2", "V3", "V4", "V14"]
    for idx, name in enumerate(quick_fields):
        with cols[idx % 4]:
            default_val = 0.0
            v = st.number_input(name, value=float(default_val))
            inputs[name] = v

    st.markdown("**Optional:** Json dict برای سایر Vها (e.g. `{\"V5\": -0.32, \"V6\": 1.2}`)")
    json_txt = st.text_input("Extra feature JSON", value="")
    if json_txt.strip():
        try:
            extra = json.loads(json_txt)
            for k, v in extra.items():
                inputs[k] = float(v)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    # به دیتافریم تبدیل و نظم ستون‌ها را حفظ کنیم
    df1 = pd.DataFrame([inputs])
    X1 = ensure_column_order(df1, feature_cols)
    if st.button("Predict"):
        try:
            proba = float(predict_proba(model, X1)[0])
            pred = int(proba >= use_thr)
            st.metric("Fraud probability", f"{proba:.4f}")
            st.metric("Predicted class", pred)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----- Batch CSV -----
with tabs[1]:
    st.subheader("Batch Inference (CSV)")
    st.caption("CSV با همان ستون‌های آموزش. اگر `Class` هم داشته باشد، متریک‌ها نمایش داده می‌شود.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview", df.head())
        has_label = "Class" in df.columns
        X = df.drop(columns=["Class"]) if has_label else df.copy()
        X = ensure_column_order(X, feature_cols)

        try:
            probs = predict_proba(model, X)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        else:
            preds = (probs >= use_thr).astype(int)
            out = df.copy()
            out["proba_fraud"] = probs
            out["pred_class"] = preds

            left, right = st.columns([2,1])
            with left:
                st.write("Predictions (first 100):")
                st.dataframe(out.head(100))
            with right:
                plot_hist(probs, use_thr)

            # Save & download
            PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(PRED_OUT, index=False, encoding="utf-8")
            st.download_button(
                label="Download predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
            st.success(f"Saved: {PRED_OUT}")

            # Metrics if label exists
            if has_label:
                y = df["Class"].astype(int).values
                try:
                    roc = roc_auc_score(y, probs)
                    ap = average_precision_score(y, probs)
                except Exception:
                    roc, ap = np.nan, np.nan
                st.markdown(f"**ROC-AUC:** {roc:.4f} | **PR-AUC (AP):** {ap:.4f}")

                st.markdown("**Classification report (with current threshold):**")
                rep = classification_report(y, preds, output_dict=False, zero_division=0)
                st.code(rep)

                st.markdown("**Confusion Matrix:**")
                show_confusion(y, preds)
