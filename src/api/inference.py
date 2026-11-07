import joblib
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from .schemas import HousePredictionRequest, PredictionResponse

MODEL_PATH = "models/trained/house_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"

try:
    print(f"🔍 Looking for model at: {MODEL_PATH}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Files in current directory: {os.listdir('.')}")
    
    # Verificar si el directorio models existe
    if os.path.exists("models/trained"):
        print(f"✅ models/trained directory exists")
        print(f"📁 Files in models/trained: {os.listdir('models/trained')}")
    else:
        print("❌ models/trained directory does not exist")
    
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"✅ Model loaded: {type(model).__name__}")
    print(f"✅ Preprocessor loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or preprocessor: {str(e)}")
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    try:
        # ✅ PASO 1: Crear DataFrame con TODOS los datos
        input_data = pd.DataFrame([{
            'sqft': request.sqft,
            'bedrooms': request.bedrooms,
            'bathrooms': request.bathrooms,
            'location': request.location,
            'year_built': request.year_built,
            'condition': request.condition,
            'price_per_sqft': request.price_per_sqft
        }])
        
        # ✅ PASO 2: Calcular features derivadas
        current_year = datetime.now().year
        input_data['house_age'] = current_year - input_data['year_built']
        input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
        
        # ✅ PASO 3: Aplicar preprocessing
        processed_features = preprocessor.transform(input_data)
        
        # ✅ PASO 4: Hacer predicción
        predicted_price = model.predict(processed_features)[0]
        
        # Convert to Python float, clamp to non-negative, and round
        predicted_price = max(0.0, float(predicted_price))
        predicted_price = round(predicted_price, 2)

        # Confidence interval (10% range), clamped to non-negative
        confidence_interval = [
            round(max(0.0, predicted_price * 0.9), 2),
            round(predicted_price * 1.1, 2)
        ]
        
        return PredictionResponse(
            predicted_price=predicted_price,
            confidence_interval=confidence_interval,
            features_importance={},
            prediction_time=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Prediction failed: {str(e)}")

def batch_predict(requests: list[HousePredictionRequest]) -> list[PredictionResponse]:
    """
    Perform batch predictions.
    """
    return [predict_price(req) for req in requests]