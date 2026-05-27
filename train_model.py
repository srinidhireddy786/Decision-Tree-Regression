import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# =========================
# Load Dataset
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

# =========================
# Model
# =========================
model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# =========================
# Save Model
# =========================
joblib.dump(model, "models/decision_tree_regressor.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model Saved Successfully")