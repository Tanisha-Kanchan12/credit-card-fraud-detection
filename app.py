import streamlit as st
import numpy as np
import pickle

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------------------------------
# App UI
# -------------------------------------------------
st.title("💳 Credit Card Fraud Detection System")

st.write(
    "This application demonstrates how a Machine Learning model can be used "
    "to detect **fraudulent credit card transactions**.\n\n"
    "⚠️ Note: All inputs are **anonymized PCA features**, similar to the original dataset."
)

st.divider()

st.subheader("🧾 Enter Transaction Details")

# -------------------------------------------------
# User Inputs (PCA-like features)
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Time", min_value=0.0, value=10000.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)

with col2:
    v4 = st.number_input("V4", value=0.0)
    v5 = st.number_input("V5", value=0.0)
    v6 = st.number_input("V6", value=0.0)
    amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Check Transaction"):
    input_data = np.array([
        time, v1, v2, v3, v4, v5, v6, amount
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected\n\nFraud Probability: **{probability:.2f}**")
    else:
        st.success(f"✅ Normal Transaction\n\nFraud Probability: **{probability:.2f}**")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption(
    "This project is built for **educational purposes** only and demonstrates "
    "an end-to-end Machine Learning workflow."
)

