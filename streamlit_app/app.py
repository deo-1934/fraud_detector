"""Simple Streamlit demo for interactive scoring."""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("اختلال‌یاب  Credit Card Fraud Demo")

# Resolve project root based on this file location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"

if not MODEL_PATH.exists():
    st.warning("Model not found. Please run `python main.py` first to train and save the model.")
else:
    pipe = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully.")

uploaded = st.file_uploader("Upload a CSV containing columns: Time, V1..V28, Amount", type=["csv"])
if uploaded and MODEL_PATH.exists():
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20))

    try:
        proba = pipe.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)

        out = df.copy()
        out["fraud_proba"] = proba
        out["fraud_pred"] = pred

        st.subheader("Predictions")
        st.dataframe(out.head(100))
        st.download_button(
            "Download predictions.csv",
            out.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
