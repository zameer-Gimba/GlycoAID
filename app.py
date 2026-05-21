# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

df = pd.read_csv(url)

# Fix invalid zero values
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

# Decision Tree Model
tree_model = DecisionTreeClassifier()

tree_model.fit(X_train, y_train)

# Predictions
log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# Accuracy
log_acc = accuracy_score(y_test, log_pred)
tree_acc = accuracy_score(y_test, tree_pred)

print("Logistic Accuracy:", log_acc)
print("Tree Accuracy:", tree_acc)

# Select best model
best_model = log_model if log_acc > tree_acc else tree_model

# App title
st.set_page_config(page_title="GlycoAID", page_icon="🩺")

st.title("🩺 GlycoAID – Diabetes Risk Prediction System")

st.write("Enter patient details below.")

# User inputs
pregnancies = st.number_input("Pregnancies", min_value=0)

glucose = st.number_input("Glucose", min_value=0.0)

blood_pressure = st.number_input("Blood Pressure", min_value=0.0)

skin_thickness = st.number_input("Skin Thickness", min_value=0.0)

insulin = st.number_input("Insulin", min_value=0.0)

bmi = st.number_input("BMI", min_value=0.0)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)

age = st.number_input("Age", min_value=1)

# Predict button
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

    # Make prediction
    prediction = model.predict(input_data)

    # Probability score
    probability = model.predict_proba(input_data)

    # Display result
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
