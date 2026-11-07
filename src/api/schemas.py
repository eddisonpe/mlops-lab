from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal

class HousePredictionRequest(BaseModel):
    sqft: float = Field(..., gt=1000, lt=5000, description="Square footage of the house")
    bedrooms: int = Field(..., ge=1, le=6, description="Number of bedrooms")
    bathrooms: float = Field(..., gt=0.5, le=5.0, description="Number of bathrooms")
    location: Literal["Rural", "Suburb", "Urban", "Downtown", "Waterfront", "Mountain"] = Field(..., description="Location type")
    year_built: int = Field(..., ge=1945, le=2023, description="Year the house was built")
    condition: Literal["Poor", "Fair", "Good", "Excellent"] = Field(..., description="Condition of the house")
    price_per_sqft: float = Field(..., gt=50, lt=1000, description="Expected price per square foot in your area (e.g., 320)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sqft": 1380,
                "bedrooms": 2,
                "bathrooms": 1.0,
                "location": "Rural",
                "year_built": 1948,
                "condition": "Poor",
                "price_per_sqft": 182
            }
        }
    )

class BatchPredictionRequest(BaseModel):
    requests: List[HousePredictionRequest] = Field(..., description="List of houses to predict prices for")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "requests": [
                    {
                        "sqft": 1380,
                        "bedrooms": 2,
                        "bathrooms": 1.0,
                        "location": "Rural",
                        "year_built": 1948,
                        "condition": "Poor",
                        "price_per_sqft": 182
                    },
                    {
                        "sqft": 1350,
                        "bedrooms": 2,
                        "bathrooms": 1.0,
                        "location": "Rural",
                        "year_built": 1947,
                        "condition": "Poor",
                        "price_per_sqft": 184
                    }
                ]
            }
        }
    )

class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Predicted house price in dollars")
    confidence_interval: List[float] = Field(..., description="90% confidence interval [lower, upper]")
    features_importance: dict = Field(default={}, description="Feature importance scores")
    prediction_time: str = Field(..., description="Timestamp of the prediction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_price": 489650.75,
                "confidence_interval": [440685.68, 538615.82],
                "features_importance": {},
                "prediction_time": "2025-07-24T12:30:45.123456"
            }
        }
    )