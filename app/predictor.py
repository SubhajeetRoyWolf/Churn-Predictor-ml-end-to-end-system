import joblib
import numpy as np

import os

# Get the directory where THIS script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the path relative to the project root
# Instead of C:/Users/..., use this:
model_path = os.path.join(BASE_DIR, "model", "olist_pipeline.pkl")

# Load the model
model = joblib.load(model_path)

def predict_churn(data):

    features = np.array([
        data.price,
        data.freight_value,
        data.delivery_time
    ]).reshape(1,-1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
	"prediction":int(prediction),"probability":float(probability)
	}