from fastapi import FastAPI
from app.schema import PredictionInput
from app.predictor import predict_churn
from app.logger import logger

app = FastAPI(
    title="Olist ML Prediction API",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict")

def predict(data: PredictionInput):

    logger.info(f"Incoming request: {data}")

    result = predict_churn(data)

    logger.info(f"Prediction result: {result}")

    return result
