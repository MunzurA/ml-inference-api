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
    # Load iris dataset
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    return X, y


def build_preprocessing():
    # Initialize scaler for feature normalization
    return StandardScaler()


def train():
    X, y = load_data()
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features to normalize values
    scaler = build_preprocessing()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)

    # Evaluate model on test set
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    logging.info(f"Model accuracy: {acc:.4f}")

    return model, scaler


def save_artifacts(model, scaler):
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Save model and scaler for later use
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, PREPROCESSOR_PATH)
    logging.info(f"Model and preprocessor saved to {MODEL_DIR}")


if __name__ == "__main__":
    # Train model and save artifacts
    model, scaler = train()
    save_artifacts(model, scaler)
