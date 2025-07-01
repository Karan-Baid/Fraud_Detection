# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# -----------------------------
# 🔧 Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_model():
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.load_model("xgb_fraud_model.json")
    return model

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# -----------------------------
# 🚀 App Layout
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.markdown("Upload a transaction CSV or enter values manually to detect fraud.")

# -----------------------------
# 📁 CSV Upload Section
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if 'Amount' in data.columns and 'Time' in data.columns:
        try:
            # ✅ Scale Amount & Time
            scaled = scaler.transform(data[['Amount', 'Time']].values)
            data['scaled_amount'] = scaled[:, 0]
            data['scaled_time'] = scaled[:, 1]
            data.drop(['Amount', 'Time'], axis=1, inplace=True)

            # ✅ Drop Class column if present
            if 'Class' in data.columns:
                data.drop('Class', axis=1, inplace=True)

            # ✅ Predict
            prediction = model.predict(data)
            prediction_proba = model.predict_proba(data)[:, 1]
            data['Fraud?'] = prediction
            data['Fraud Probability'] = prediction_proba

            st.success("✅ Predictions generated successfully.")
            st.dataframe(data)

        except Exception as e:
            st.error(f"Error during prediction: {e}")


# -----------------------------
# 🧾 Manual Entry Section
# -----------------------------
st.markdown("---")
st.header("📋 Manual Entry")

# Build input form
input_features = {}
input_features['Amount'] = st.number_input("Transaction Amount", value=100.0)
input_features['Time'] = st.number_input("Transaction Time (seconds since first transaction)", value=50000)

for i in range(1, 29):
    input_features[f'V{i}'] = st.number_input(f'V{i}', value=0.0)

if st.button("🔍 Predict Fraud"):
    try:
        # ✅ Scale Amount and Time
        scaled_vals = scaler.transform([[input_features['Amount'], input_features['Time']]])
        scaled_amount = scaled_vals[0][0]
        scaled_time = scaled_vals[0][1]

        # Build input DataFrame
        input_data = {
            'scaled_amount': [scaled_amount],
            'scaled_time': [scaled_time]
        }
        for i in range(1, 29):
            input_data[f'V{i}'] = [input_features[f'V{i}']]

        input_df = pd.DataFrame(input_data)

        # ✅ Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.markdown(f"**Fraud Probability:** `{prob:.4f}`")
        if pred == 1:
            st.error("⚠️ Fraud Detected!")
        else:
            st.success("✅ Transaction is likely legitimate.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
