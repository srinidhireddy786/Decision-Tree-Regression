import streamlit as st
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Decision Tree Regression",
    layout="wide"
)

# =========================
# CSS
# =========================
st.markdown("""
<style>

h1 {
    text-align: center;
    color: #7b241c;
    font-size: 45px !important;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Dataset
# =========================
data = load_diabetes()

X = data.data
y = data.target

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Load Model
# =========================
model = joblib.load("models/decision_tree_regressor.pkl")

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# Metrics
# =========================
mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

# =========================
# Title
# =========================
st.title("Decision Tree Regression")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Input Features")

features = []

for i in range(10):

    value = st.sidebar.number_input(
        f"Feature {i+1}",
        value=0.05
    )

    features.append(value)

# =========================
# Prediction
# =========================
if st.sidebar.button("Predict"):

    input_data = np.array([features])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Value: {prediction[0]:.2f}"
    )

# =========================
# Statistics
# =========================
st.subheader("Model Statistics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "MSE",
    f"{mse:.2f}"
)

col2.metric(
    "MAE",
    f"{mae:.2f}"
)

col3.metric(
    "R2 Score",
    f"{r2:.2f}"
)

# =========================
# Prediction Table
# =========================
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.subheader("Prediction Results")

st.dataframe(results.head(20))