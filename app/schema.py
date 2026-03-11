from pydantic import BaseModel

class PredictionInput(BaseModel):
    
    price: float
    freight_value: float
    delivery_time: float