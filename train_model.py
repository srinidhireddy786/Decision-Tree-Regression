# train_model.py

# =========================
# Import Libraries
# =========================
import joblib

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
# Load Dataset
# =========================
data = fetch_california_housing()

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
# Feature Scaling
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Decision Tree Regressor
# =========================
model = DecisionTreeRegressor(
    max_depth=8,
    random_state=42
)

# =========================
# Train Model
# =========================
model.fit(X_train_scaled, y_train)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test_scaled)

# =========================
# Evaluation Metrics
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
# Print Metrics
# =========================
print("\nModel Performance\n")

print(f"Mean Squared Error : {mse:.2f}")

print(f"Mean Absolute Error : {mae:.2f}")

print(f"R2 Score : {r2:.2f}")

# =========================
# Save Model & Scaler
# =========================
joblib.dump(
    model,
    "models/decision_tree_regressor.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel and Scaler Saved Successfully")