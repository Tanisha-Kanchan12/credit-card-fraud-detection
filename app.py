import streamlit as st
import pickle
import numpy as np

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ------------------ Load Model ------------------
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# ------------------ UI ------------------
st.title("💳 Credit Card Fraud Detection System")

st.write(
    "This application demonstrates how a Machine Learning model can "
    "identify potentially fraudulent credit card transactions."
)

st.divider()

st.subheader("🔍 Fraud Detection Demo")

st.write(
    "For demonstration purposes, the model evaluates a **simulated transaction** "
    "using anonymized numerical features."
)

# ------------------ Button ------------------
if st.button("🚀 Check Transaction Risk"):
    
    # Random dummy input (same shape as training features)
    input_data = np.random.normal(0, 1, (1, model.coef_.shape[1]))
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Model Prediction")

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected\n\nRisk Probability: {probability:.2f}")
    else:
        st.success(f"✅ Transaction is Normal\n\nRisk Probability: {probability:.2f}")

st.divider()

st.caption(
    "⚠️ Disclaimer: This is a demonstration project created for learning purposes. "
    "All data used is anonymized and does not represent real customer information."
)
