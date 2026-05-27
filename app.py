# app.py

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="California Housing Price Prediction",
    layout="wide"
)

# =========================
# Advanced CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #f4f6f7;
}

h1 {
    text-align: center;
    color: #154360;
    font-size: 52px !important;
    font-weight: bold;
}

h2, h3 {
    color: #1b4f72;
}

.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 22px !important;
    border-radius: 12px;
    background-color: #1f618d;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #154360;
    color: white;
}

div[data-baseweb="input"] input {
    font-size: 18px !important;
    border-radius: 10px;
}

[data-testid="stMetricValue"] {
    font-size: 30px;
    color: #117864;
}

[data-testid="stDataFrame"] {
    font-size: 17px;
}

.css-1d391kg {
    background-color: #d6eaf8;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Load Dataset
# =========================
data = fetch_california_housing()

X = data.data
y = data.target

feature_names = data.feature_names

# =========================
# DataFrame
# =========================
df = pd.DataFrame(
    X,
    columns=feature_names
)

df["Target"] = y

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
# Model
# =========================
model = DecisionTreeRegressor(
    max_depth=8,
    random_state=42
)

model.fit(X_train_scaled, y_train)

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
st.title("California Housing Price Prediction")

st.write("""
This Decision Tree Regression app predicts California housing prices
using the preloaded California Housing Dataset from Scikit-learn.
""")

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Enter Housing Features")

inputs = []

for feature in feature_names:

    value = st.sidebar.number_input(
        feature,
        value=float(df[feature].mean()),
        format="%.2f"
    )

    inputs.append(value)

# =========================
# Prediction
# =========================
if st.sidebar.button("Predict House Price"):

    input_data = np.array([inputs])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted House Price: ${prediction[0] * 100000:.2f}"
    )

# =========================
# Statistics
# =========================
st.subheader("Model Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MSE",
        f"{mse:.2f}"
    )

with col2:
    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

with col3:
    st.metric(
        "R2 Score",
        f"{r2:.2f}"
    )

# =========================
# Dataset Preview
# =========================
st.subheader("Dataset Preview")

st.dataframe(df.head(20))

# =========================
# Prediction Results
# =========================
results = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

st.subheader("Prediction Results")

st.dataframe(results.head(20))

# =========================
# Feature Importance
# =========================
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.subheader("Feature Importance")

st.dataframe(importance_df)

# =========================
# Footer
# =========================
st.markdown("""
---
### Developed with Streamlit & Scikit-learn
""")