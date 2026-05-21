import streamlit as st
import numpy as np
import joblib

# App title
st.set_page_config(page_title="GlycoAID", page_icon="🩺")

st.title("🩺 GlycoAID – Diabetes Risk Prediction System")

st.write("Enter patient medical details below.")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0)

glucose = st.number_input("Glucose", min_value=0.0)

blood_pressure = st.number_input("Blood Pressure", min_value=0.0)

skin_thickness = st.number_input("Skin Thickness", min_value=0.0)

insulin = st.number_input("Insulin", min_value=0.0)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=1)

# Prediction button
if st.button("Predict Diabetes Risk"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)

    # Output
    if prediction[0] == 1:

        st.error("🚨 High Risk of Diabetes")

        st.write(f"Risk Probability: {probability[0][1] * 100:.2f}%")

    else:

        st.success("✅ Low Risk of Diabetes")

        st.write(f"Risk Probability: {probability[0][0] * 100:.2f}%")

# Disclaimer
st.warning(
    "This system is for educational purposes only and does not replace professional medical diagnosis."
)
