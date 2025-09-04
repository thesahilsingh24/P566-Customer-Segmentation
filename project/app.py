import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Segmentation — Predictor", layout="centered")
st.title("Customer Segmentation — KMeans Predictor")

# --- Load artifacts ---
try:
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

try:
    with open("features.json", "r") as f:
        FEATURES = json.load(f)
    if not isinstance(FEATURES, list) or len(FEATURES) == 0:
        raise ValueError("features.json must contain a non-empty list of feature names.")
except Exception as e:
    st.error(f"Failed to load features.json: {e}")
    st.stop()

st.write("Enter the feature values below. The form is built from your features.json, so it matches training exactly.")

# --- Input form ---
with st.form("predict_form"):
    values = {}
    for feat in FEATURES:
        # sensible defaults per field name heuristics
        default = 0.0
        step = 1.0
        fl = feat.lower()
        if fl.startswith("mnt"):
            default, step = 100.0, 10.0
        elif "income" in fl:
            default, step = 50000.0, 1000.0
        elif "recency" in fl:
            default, step = 30.0, 1.0
        elif "purchases" in fl or "visits" in fl:
            default, step = 1.0, 1.0
        elif "total_spending" in fl or "total_spent" in fl:
            default, step = 1000.0, 10.0

        values[feat] = st.number_input(feat, value=float(default), step=float(step))

    submitted = st.form_submit_button("Predict segment")

if submitted:
    try:
        X = pd.DataFrame([values])[FEATURES]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        st.success(f"Predicted Segment: {int(pred[0])}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
