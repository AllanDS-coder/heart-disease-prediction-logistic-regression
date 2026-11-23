
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Define the features used by the model

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
]


FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "num"

def build_model():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    return model_pipeline

def prepare_data(data):

    data = data.copy()

    # Remove identifiers and study location
    data = data.drop(
        columns=["id", "dataset", "ca", "thal"],
        errors="ignore",
    )

    # Convert original target into binary classification
    data[TARGET] = (data[TARGET] > 0).astype(int)

    X = data[FEATURES]
    y = data[TARGET]

    return X, y

def save_model(model, destination="models/heart_disease_model.joblib"):

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, destination)

    print(f"Model saved to: {destination}")

def load_model(model_path="models/heart_disease_model.joblib"):

    return joblib.load(model_path)