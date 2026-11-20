
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------
# 1. Create FastAPI application
# --------------------------------

app = FastAPI(
    title="Crop Prediction API",
    description="Predict suitable crops using soil and weather data",
    version="1.0"
)

# --------------------------------
# 2. Load trained ML model
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "crop_model.pkl not found. Train the model first."
    )

model = joblib.load(MODEL_PATH)

# --------------------------------
# 3. Define input data
# --------------------------------

class CropInput(BaseModel):

    N: float = Field(ge=0)
    P: float = Field(ge=0)
    K: float = Field(ge=0)

    temperature: float = Field(ge=-50, le=70)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0)


# --------------------------------
# 4. Home route
# --------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to Crop Prediction API",
        "status": "API is running"
    }


# --------------------------------
# 5. Crop prediction route
# --------------------------------

@app.post("/predict")
def predict_crop(data: CropInput):

    try:

        input_data = pd.DataFrame([{
            "N": data.N,
            "P": data.P,
            "K": data.K,
            "temperature": data.temperature,
            "humidity": data.humidity,
            "ph": data.ph,
            "rainfall": data.rainfall
        }])

        # Match the feature order used during training
        input_data = input_data[model.feature_names_in_]

        prediction = model.predict(input_data)

        return {
            "predicted_crop": str(prediction[0]),
            "message": "Crop predicted successfully"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
