import streamlit as st
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ---------------- Load Model ----------------
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- UI ----------------
st.title("💳 Credit Card Fraud Detection System")

st.write(
    "This application demonstrates how a Machine Learning model evaluates "
    "credit card transactions and flags potentially fraudulent activity."
)

st.divider()

st.subheader("🔍 Transaction Risk Analysis")

st.write(
    "The model internally uses anonymized numerical features derived from "
    "transaction behavior. Below is a **summary view** similar to what "
    "risk analysts see in real systems."
)

# ---------------- Button ----------------
if st.button("🚀 Analyze Transaction"):

    # Simulated transaction (same feature count as training data)
    input_data = np.random.normal(0, 1, (1, model.coef_.shape[1]))

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Fake but realistic metadata for display
    amount = np.random.uniform(10, 5000)
    time_sec = np.random.randint(0, 172800)

    st.subheader("📄 Transaction Summary")

    col1, col2 = st.columns(2)
    col1.metric("Transaction Amount (₹)", f"{amount:.2f}")
    col2.metric("Transaction Time (sec)", time_sec)

    st.subheader("📊 Model Decision")

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected\n\nRisk Probability: {probability:.2f}")
    else:
        st.success(f"✅ Transaction is Normal\n\nRisk Probability: {probability:.2f}")

st.divider()

st.caption(
    "⚠️ Disclaimer: This is a demonstration project created for learning purposes. "
    "The original dataset contains confidential, anonymized features which are "
    "not exposed in the user interface."
)
