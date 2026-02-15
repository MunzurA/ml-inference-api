import joblib
import logging
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import IrisFeatures

app = FastAPI()

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessing.pkl"


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError("Model artifacts not found. Run training first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(PREPROCESSOR_PATH)
    return model, scaler


model, scaler = load_artifacts()


@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        input_data = np.array(
            [
                [
                    features.feature1,
                    features.feature2,
                    features.feature3,
                    features.feature4,
                ]
            ]
        )

        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        probability = float(np.max(model.predict_proba(scaled_data)))

        return {"prediction": int(prediction), "probability": probability}

    except Exception as e:
        logging.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Inference failed")
