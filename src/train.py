import os
import joblib
import logging
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessing.pkl")


def load_data():
    data = load_iris(as_frame=True)
    x = data.data
    y = data.target
    return x, y


def build_preprocessing():
    return StandardScaler()


def train():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = build_preprocessing()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    logging.info(f"Model accuracy: {acc:.4f}")

    return model, scaler


def save_artifacts(model, scaler):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, PREPROCESSOR_PATH)
    logging.info(f"Model and preprocessor saved to {MODEL_DIR}")


if __name__ == "__main__":
    model, scaler = train()
    save_artifacts(model, scaler)
