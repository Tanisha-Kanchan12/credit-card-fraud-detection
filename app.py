import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Load dataset (for demo)
# -----------------------------
data = pd.read_csv("creditcard.csv")

# -----------------------------
# App UI
# -----------------------------
st.title("💳 Credit Card Fraud Detection")
st.write(
    "This is a **demo web application** that shows how a Machine Learning model "
    "can be used to detect fraudulent credit card transactions."
)

st.write(
    "Since the dataset uses anonymized features (PCA), the app works by "
    "testing the model on a **random real transaction** from the dataset."
)

st.divider()

# -----------------------------
# Button action
# -----------------------------
if st.button("🔄 Check Random Transaction"):
    sample = data.sample(1).reset_index(drop=True)

    X_sample = sample.drop(columns=["Class"])
    actual_label = int(sample["Class"].iloc[0])

    prediction = model.predict(X_sample)[0]
    probability = model.predict_proba(X_sample)[0][1]

    st.subheader("📄 Sample Transaction (Anonymized)")
    st.dataframe(sample)

    st.subheader("🔍 Model Result")
    if prediction == 1:
        st.error(f"⚠️ Fraud Detected\n\nFraud Probability: {probability:.2f}")
    else:
        st.success(f"✅ Normal Transaction\n\nFraud Probability: {probability:.2f}")

    st.caption(f"Actual label in dataset (for demo): {actual_label}")

st.divider()

# -----------------------------
# Footer note
# -----------------------------
st.caption(
    "Note: This application is for educational purposes only. "
    "It demonstrates an end-to-end Machine Learning pipeline using anonymized data."
)
