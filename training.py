# Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset from GitHub
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

df = pd.read_csv(url)

# Replace invalid zeros with NaN
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

# Decision Tree
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

# Save model and scaler
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
