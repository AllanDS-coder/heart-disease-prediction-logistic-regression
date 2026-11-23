
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)

from model import build_model, prepare_data, save_model


# Project root
ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = ROOT / "data" / "heart.csv"

# Load dataset
data = pd.read_csv(DATA_PATH)

# Prepare data
X, y = prepare_data(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# Build model
model = build_model()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nMODEL EVALUATION")
print("-----------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Log Loss:", log_loss(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
save_model(
    model,
    ROOT / "models" / "heart_disease_model.joblib",
)

print("\nTraining completed successfully.")
