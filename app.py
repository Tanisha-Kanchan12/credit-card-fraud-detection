st.subheader("🧾 Enter Transaction Details")

time = st.number_input("Time (seconds since first transaction)", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

st.write("PCA Features (Anonymized)")
v1 = st.number_input("V1")
v2 = st.number_input("V2")
v3 = st.number_input("V3")
v4 = st.number_input("V4")

if st.button("🔍 Check Transaction"):
    input_data = [[time, amount, v1, v2, v3, v4]]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction\n\nFraud Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legit Transaction\n\nFraud Probability: {probability:.2f}")
