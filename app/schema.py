from pydantic import BaseModel
from typing import Optional


class PredictionInput(BaseModel):
    price: float
    freight_value: float
    delivery_time: float
    # Optional engineered/source signal
    is_delayed: Optional[bool] = None
