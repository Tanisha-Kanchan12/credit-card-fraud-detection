import streamlit as st
import numpy as np
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------------- UI ----------------
st.title("💳 Credit Card Fraud Detection System")

st.markdown(
"""
This system simulates how **banks detect suspicious credit card transactions**
in real time using Machine Learning.

The transaction details shown below are **internally processed risk signals**
and are not visible to customers in real banking systems.
"""
)

st.divider()
st.subheader("🧾 Transaction Risk Parameters")

# ---------------- Generate Inputs ----------------
# Model expects SAME number of features it was trained on
n_features = model.coef_.shape[1]

inputs = []
for i in range(n_features):
    value = st.number_input(
        f"Feature {i+1}",
        value=0.0,
        step=0.01
    )
    inputs.append(value)

# ---------------- Prediction ----------------
if st.button("🔍 Analyze Transaction"):
    X = np.array(inputs).reshape(1, -1)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    st.divider()
    st.subheader("🔎 Risk Assessment Result")

    if prediction == 1:
        st.error(
            f"⚠️ High Risk Transaction Detected\n\n"
            f"Fraud Probability: **{probability:.2f}**"
        )
    else:
        st.success(
            f"✅ Transaction Approved\n\n"
            f"Fraud Probability: **{probability:.2f}**"
        )

# ---------------- Footer ----------------
st.divider()
st.caption(
    "Disclaimer: This application is a simulation built for academic and learning purposes."
)

