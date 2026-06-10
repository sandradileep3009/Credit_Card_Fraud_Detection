import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Credit Card Fraud Detection",page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount",min_value=0.0,value=100.0)

time = st.number_input("Transaction Time",min_value=0.0, value=0.)

features = []

for i in range(1, 29):
    value = st.number_input(f"V{i}",value=0.0,format="%.6f")
    features.append(value)

if st.button("Predict Fraud")
    data = [[time,*features,amount]]
    columns = ["Time",*[f"V{i}" for i in range(1, 29)],"Amount"]
    df = pd.DataFrame(data,columns=columns)
    df["Amount"] = scaler.transform(df[["Amount"]])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction\n\nProbability: {probability:.2%}")
