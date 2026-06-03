"""Churn predictor — loads tuned RandomForest with engineered features."""
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "olist_pipeline.pkl")

# Load once at import — single artifact contains model + feature order
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
FEATURES = artifact["features"]


def predict_churn(data):
    """Build engineered features from raw inputs, then predict."""
    price = data.price
    freight = data.freight_value
    delivery = data.delivery_time
    is_delayed = (
        int(data.is_delayed) if getattr(data, "is_delayed", None) is not None
        else int(delivery > 15)
    )

    row = {
        "price": price,
        "freight_value": freight,
        "delivery_time": delivery,
        "total_cost": price + freight,
        "freight_ratio": freight / (price + 1),
        "is_expensive": int(price > 100),
        "is_delayed": is_delayed,
    }
    X = pd.DataFrame([row])[FEATURES]
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    return {"prediction": int(prediction), "probability": float(probability)}
