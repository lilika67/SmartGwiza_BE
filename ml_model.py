import joblib
import pandas as pd
import requests
import io
import tempfile
import os
from fastapi import HTTPException
from config import GOOGLE_DRIVE_FILE_ID, GOOGLE_DRIVE_URL

# Global variables
model_artifacts = None
model = None
scaler = None
feature_names = []


def load_model_from_google_drive():
    """Load model directly from Google Drive"""
    try:
        print(" Loading model from Google Drive...")
        session = requests.Session()
        response = session.get(GOOGLE_DRIVE_URL, stream=True)

        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            params = {"id": GOOGLE_DRIVE_FILE_ID, "confirm": token}
            response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)

        if response.status_code != 200:
            raise Exception(
                f"Failed to download model. Status code: {response.status_code}"
            )

        model_content = io.BytesIO(response.content)
        model_artifacts = joblib.load(model_content)

        print(" ML Model loaded successfully from Google Drive!")
        return model_artifacts
    except Exception as e:
        print(f" Error loading model from Google Drive: {e}")
        return None


def load_model_from_google_drive_with_tempfile():
    """Load model by downloading to temporary file"""
    try:
        print(" Downloading model from Google Drive to temporary file...")
        session = requests.Session()
        response = session.get(GOOGLE_DRIVE_URL, stream=True)

        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            params = {"id": GOOGLE_DRIVE_FILE_ID, "confirm": token}
            response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_path = temp_file.name

        model_artifacts = joblib.load(temp_path)
        os.unlink(temp_path)

        print(" ML Model loaded successfully via temporary file!")
        return model_artifacts
    except Exception as e:
        print(f" Error loading model: {e}")
        return None


def predict_yield(input_data):
    """Make prediction using the trained model"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction model not available. Please try again later.",
        )

    try:
        input_dict = {}
        input_dict["rainfall_mm"] = [input_data.rainfall_mm]
        input_dict["temperature_c"] = [input_data.temperature_c]
        input_dict["soil_ph"] = [input_data.soil_ph]
        input_dict["fertilizer_used_kg_per_ha"] = [input_data.fertilizer_used_kg_per_ha]
        input_dict["pesticide_l_per_ha"] = [input_data.pesticide_l_per_ha]
        input_dict["irrigation_type_encoded"] = [
            1 if input_data.irrigation_type == "Rainfed" else 0
        ]

        for feature in feature_names:
            if feature.startswith("district_"):
                district_feature_name = feature.replace("district_", "")
                input_dict[feature] = [
                    1 if district_feature_name == input_data.district else 0
                ]

        input_df = pd.DataFrame(input_dict)

        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)
        predicted_yield = model.predict(input_scaled)[0]

        return float(predicted_yield)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction processing error: {str(e)}"
        )


def get_interpretation(yield_value: float) -> dict:
    """Get interpretation based on yield value"""
    if yield_value >= 7.0:
        return {"interpretation": "Excellent yield potential", "confidence": "high"}
    elif yield_value >= 6.0:
        return {"interpretation": "Good yield potential", "confidence": "medium"}
    elif yield_value >= 5.0:
        return {"interpretation": "Average yield potential", "confidence": "medium"}
    else:
        return {"interpretation": "Low yield potential", "confidence": "low"}


def initialize_model():
    """Initialize the ML model"""
    global model_artifacts, model, scaler, feature_names

    model_artifacts = load_model_from_google_drive()
    if not model_artifacts:
        print(" Trying alternative loading method...")
        model_artifacts = load_model_from_google_drive_with_tempfile()

    if model_artifacts:
        model = model_artifacts.get("model")
        scaler = model_artifacts.get("scaler")
        feature_names = model_artifacts.get("feature_names", [])
        print(f" Model loaded successfully! Features: {len(feature_names)}")
        return True
    else:
        print(" Could not load ML model - predictions will not be available")
        return False
