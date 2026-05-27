# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Decision Tree Regression App",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
# =========================
# Enhanced Custom CSS
# =========================
st.markdown("""
<style>

/* Main Background */
.main {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

/* Title Styling */
h1 {
    text-align: center;
    color: #0b3c5d;
    font-size: 56px !important;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 10px;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.15);
}

/* Subheaders */
h2, h3 {
    color: #145374;
    font-weight: 700;
    margin-top: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #154360, #1f618d);
    padding-top: 20px;
}

/* Sidebar Text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .css-1d391kg,
section[data-testid="stSidebar"] p {
    color: white !important;
    font-size: 18px !important;
}

/* Input Boxes */
div[data-baseweb="input"] input {
    background-color: #f8f9f9;
    border-radius: 12px;
    border: 2px solid #d6dbdf;
    padding: 10px;
    font-size: 18px !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 22px !important;
    font-weight: bold;
    border-radius: 14px;
    background: linear-gradient(to right, #1f618d, #3498db);
    color: white;
    border: none;
    transition: 0.3s ease;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}

/* Button Hover Effect */
.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right, #154360, #2471a3);
    color: white;
}

/* Metrics */
[data-testid="metric-container"] {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.12);
    border-left: 8px solid #1f618d;
}

[data-testid="stMetricValue"] {
    font-size: 32px;
    color: #117864;
    font-weight: bold;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid #d5d8dc;
    font-size: 17px;
}

/* Expanders */
.streamlit-expanderHeader {
    font-size: 20px !important;
    font-weight: bold;
    color: #154360 !important;
}

/* Success Message */
.stAlert {
    border-radius: 12px;
    font-size: 18px;
}

/* Footer */
footer {
    visibility: hidden;
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
# Load Saved Model
# =========================
model = joblib.load(
    "models/decision_tree_regressor.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

# =========================
# Scale Data
# =========================
X_test_scaled = scaler.transform(X_test)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# Metrics
# =========================
mse = mean_squared_error(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)

# =========================
# Title
# =========================
st.title("California Housing Price Prediction")

st.write("""
This application predicts California housing prices
using Decision Tree Regression with the California Housing Dataset.
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
# Model Statistics
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

st.dataframe(df.head(15))

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
# Visualization 1
# Actual vs Predicted
# =========================
st.subheader("Actual vs Predicted Prices")

fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.scatter(
    y_test,
    y_pred
)

ax1.set_xlabel("Actual Prices")

ax1.set_ylabel("Predicted Prices")

ax1.set_title("Actual vs Predicted")

st.pyplot(fig1)

# =========================
# Visualization 2
# Feature Importance
# =========================
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax2.set_xlabel("Importance")

ax2.set_ylabel("Features")

ax2.set_title("Feature Importance")

st.pyplot(fig2)

# =========================
# Visualization 3
# Target Distribution
# =========================
st.subheader("Target Distribution")

fig3, ax3 = plt.subplots(figsize=(8, 6))

ax3.hist(
    y,
    bins=30
)

ax3.set_xlabel("House Price")

ax3.set_ylabel("Frequency")

ax3.set_title("Distribution of Target Values")

st.pyplot(fig3)

# =========================
# Footer
# =========================
st.markdown("""
---
### Built with Streamlit & Scikit-learn
""")