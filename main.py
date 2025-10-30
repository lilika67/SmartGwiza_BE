# main.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, field_validator, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from motor.motor_asyncio import AsyncIOMotorClient
import joblib
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import io
import tempfile
import re
import uuid
import hashlib
from contextlib import asynccontextmanager

load_dotenv()

# ===== CONFIG =====
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-make-it-strong")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Google Drive file ID from your URL
GOOGLE_DRIVE_FILE_ID = "162HkTkMQgMMa1Spg4l-eNZwry1_iIlcp"
GOOGLE_DRIVE_URL = (
    f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
)

# Global variables
client = None
database = None
users_collection = None
submissions_collection = None
predictions_collection = None
model_artifacts = None
model = None
scaler = None
feature_names = []


async def init_database():
    """Initialize database connection"""
    global client, database, users_collection, submissions_collection, predictions_collection
    try:
        MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        client = AsyncIOMotorClient(
          MONGODB_URL,
          tls=True,
          tlsAllowInvalidCertificates=True,  # For development only
          )
        # Test connection
        await client.admin.command("ping")
        database = client.smart_gwiza
        users_collection = database.users
        submissions_collection = database.submissions
        predictions_collection = database.predictions
        print("✅ Database connected successfully")
        return True
    except Exception as e:
        print(f" Database connection failed: {e}")
        return False


def load_model_from_google_drive():
    """Load model directly from Google Drive"""
    try:
        print(" Loading model from Google Drive...")

        # Create a session to handle cookies
        session = requests.Session()

        # First request to get the confirmation token for large files
        response = session.get(GOOGLE_DRIVE_URL, stream=True)

        # Check if we need to confirm download (for large files)
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            # Make second request with confirmation token
            params = {"id": GOOGLE_DRIVE_FILE_ID, "confirm": token}
            response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)

        # Check if request was successful
        if response.status_code != 200:
            raise Exception(
                f"Failed to download model. Status code: {response.status_code}"
            )

        # Load model directly from the response content
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

        # Handle confirmation token for large files
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            params = {"id": GOOGLE_DRIVE_FILE_ID, "confirm": token}
            response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            temp_path = temp_file.name

        # Load model from temporary file
        model_artifacts = joblib.load(temp_path)

        # Clean up temporary file
        os.unlink(temp_path)

        print(" ML Model loaded successfully via temporary file!")
        return model_artifacts

    except Exception as e:
        print(f" Error loading model: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_artifacts, model, scaler, feature_names

    print("🚀 Starting Smart Gwiza API...")

    # Initialize database connection
    db_success = await init_database()
    if not db_success:
        print("  Starting without database connection")

    # Try to load model
    model_artifacts = load_model_from_google_drive()
    if not model_artifacts:
        print(" Trying alternative loading method...")
        model_artifacts = load_model_from_google_drive_with_tempfile()

    if model_artifacts:
        model = model_artifacts.get("model")
        scaler = model_artifacts.get("scaler")
        feature_names = model_artifacts.get("feature_names", [])
        print(f" Model loaded successfully! Features: {len(feature_names)}")
    else:
        print(" Could not load ML model - predictions will not be available")

    yield
    # Shutdown
    if client:
        client.close()


# ===== FASTAPI APP =====
app = FastAPI(
    title="Smart Gwiza - Maize Yield Prediction API",
    description="AI-powered maize yield prediction for Rwandan farmers",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== AUTH SETUP =====
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


async def get_db_collections():
    """Get database collections with safety check"""
    if database is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return users_collection, submissions_collection, predictions_collection


async def verify_db_connection():
    """Verify database is available"""
    if database is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")


# ===== PYDANTIC MODELS =====
class UserSignup(BaseModel):
    fullname: str = Field(..., min_length=2, max_length=100)
    phone_number: str
    password: str = Field(..., min_length=8, max_length=50)
    role: str = "farmer"

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ["farmer", "admin"]:
            raise ValueError("Role must be farmer or admin")
        return v

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, v):
        # Check if password exceeds bcrypt's 72-byte limit when encoded
        if len(v.encode("utf-8")) > 72:
            raise ValueError(
                "Password is too long. Please use a shorter password (max ~50 characters)."
            )
        return v


class UserLogin(BaseModel):
    phone_number: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    fullname: str


class PredictionInput(BaseModel):
    district: str
    rainfall_mm: float
    temperature_c: float
    soil_ph: float
    fertilizer_used_kg_per_ha: float
    pesticide_l_per_ha: float
    irrigation_type: str

    @field_validator("rainfall_mm")
    @classmethod
    def validate_rainfall(cls, v):
        if v < 500 or v > 2000:
            raise ValueError("Rainfall should be between 500 and 2000 mm")
        return v

    @field_validator("temperature_c")
    @classmethod
    def validate_temperature(cls, v):
        if v < 15 or v > 30:
            raise ValueError("Temperature should be between 15 and 30°C")
        return v


class PredictionResponse(BaseModel):
    predicted_yield: float
    confidence: str
    interpretation: str
    prediction_id: str
    timestamp: datetime


class DataSubmission(BaseModel):
    district: str
    rainfall_mm: float
    temperature_c: float
    soil_ph: float
    fertilizer_kg_per_ha: float
    pesticide_l_per_ha: float
    irrigation_type: str
    actual_yield_tons_per_ha: float
    planting_date: Optional[str] = None
    harvest_date: Optional[str] = None
    notes: Optional[str] = None


class SubmissionResponse(BaseModel):
    success: bool
    message: str
    submission_id: str
    points_earned: int = 0


# ===== AUTH UTILITIES =====
def safe_hash_password(password: str) -> str:
    """
    Safely hash password, handling bcrypt's 72-byte limit by pre-hashing if necessary
    """
    password_bytes = password.encode("utf-8")

    # If password is within bcrypt's limit, hash directly
    if len(password_bytes) <= 72:
        return pwd_context.hash(password)

    # If password exceeds limit, pre-hash with SHA-256 first
    pre_hashed = hashlib.sha256(password_bytes).hexdigest()
    return pwd_context.hash(pre_hashed)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password, handling both direct bcrypt hashes and pre-hashed passwords
    """
    # First try direct verification
    if pwd_context.verify(plain_password, hashed_password):
        return True

    # If direct verification fails and password might be long, try pre-hashed verification
    password_bytes = plain_password.encode("utf-8")
    if len(password_bytes) > 72:
        pre_hashed = hashlib.sha256(password_bytes).hexdigest()
        return pwd_context.verify(pre_hashed, hashed_password)

    return False


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def validate_rwandan_phone(phone: str) -> str:
    phone = re.sub(r"\s+", "", phone)
    if phone.startswith("+250"):
        phone = phone
    elif phone.startswith("250"):
        phone = "+" + phone
    elif phone.startswith("0"):
        phone = "+250" + phone[1:]
    else:
        raise HTTPException(status_code=400, detail="Invalid phone number format")

    if not re.match(r"^\+250(78|79|72|73)\d{7}$", phone):
        raise HTTPException(status_code=400, detail="Invalid Rwandan phone number")

    return phone


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        phone_number: str = payload.get("sub")
        role: str = payload.get("role")
        if phone_number is None or role is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Verify database connection first
    await verify_db_connection()
    users_coll, _, _ = await get_db_collections()

    user = await users_coll.find_one({"phone_number": phone_number})
    if user is None:
        raise credentials_exception

    return user


# ===== PREDICTION FUNCTION =====
def predict_yield(input_data: PredictionInput) -> float:
    """Make prediction using the trained model"""
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction model not available. Please try again later.",
        )

    try:
        # Create input DataFrame
        input_dict = {}

        # Add numerical features
        input_dict["rainfall_mm"] = [input_data.rainfall_mm]
        input_dict["temperature_c"] = [input_data.temperature_c]
        input_dict["soil_ph"] = [input_data.soil_ph]
        input_dict["fertilizer_used_kg_per_ha"] = [input_data.fertilizer_used_kg_per_ha]
        input_dict["pesticide_l_per_ha"] = [input_data.pesticide_l_per_ha]

        # Encode irrigation type
        input_dict["irrigation_type_encoded"] = [
            1 if input_data.irrigation_type == "Rainfed" else 0
        ]

        # Add district features (one-hot encoding)
        for feature in feature_names:
            if feature.startswith("district_"):
                district_feature_name = feature.replace("district_", "")
                input_dict[feature] = [
                    1 if district_feature_name == input_data.district else 0
                ]

        # Create DataFrame and ensure correct order
        input_df = pd.DataFrame(input_dict)

        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[feature_names]

        # Scale features and predict
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


# ===== ROUTES =====
@app.get("/")
async def root():
    return {
        "message": "Smart Gwiza API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "database_connected": database is not None,
        "model_source": "Google Drive",
    }


@app.get("/health")
async def health_check():
    db_status = "disconnected"
    if database is not None:
        try:
            await database.client.admin.command("ping")
            db_status = "connected"
        except Exception:
            db_status = "disconnected"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "database": db_status,
        "model_source": "Google Drive",
    }


# Authentication routes
@app.post("/api/auth/signup", response_model=Token)
async def signup(user_data: UserSignup):
    await verify_db_connection()
    users_coll, _, _ = await get_db_collections()

    normalized_phone = validate_rwandan_phone(user_data.phone_number)

    existing_user = await users_coll.find_one({"phone_number": normalized_phone})
    if existing_user:
        raise HTTPException(status_code=400, detail="Phone number already registered")

    user_doc = {
        "fullname": user_data.fullname,
        "phone_number": normalized_phone,
        "password_hash": safe_hash_password(user_data.password),
        "role": user_data.role,
        "points": 0,
        "created_at": datetime.utcnow(),
        "last_login": datetime.utcnow(),
    }

    result = await users_coll.insert_one(user_doc)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": normalized_phone, "role": user_data.role},
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user_data.role,
        fullname=user_data.fullname,
    )


@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    await verify_db_connection()
    users_coll, _, _ = await get_db_collections()

    normalized_phone = validate_rwandan_phone(user_data.phone_number)

    user = await users_coll.find_one({"phone_number": normalized_phone})
    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid phone number or password")

    await users_coll.update_one(
        {"phone_number": normalized_phone}, {"$set": {"last_login": datetime.utcnow()}}
    )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": normalized_phone, "role": user["role"]},
        expires_delta=access_token_expires,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        role=user["role"],
        fullname=user["fullname"],
    )


# Prediction routes
@app.post("/api/predict", response_model=PredictionResponse)
async def make_prediction(
    input_data: PredictionInput, current_user: dict = Depends(get_current_user)
):
    await verify_db_connection()
    _, _, predictions_coll = await get_db_collections()

    try:
        predicted_yield = predict_yield(input_data)
        interpretation_data = get_interpretation(predicted_yield)
        prediction_id = str(uuid.uuid4())

        prediction_record = {
            "prediction_id": prediction_id,
            "user_phone": current_user["phone_number"],
            "inputs": input_data.dict(),
            "predicted_yield": predicted_yield,
            "interpretation": interpretation_data,
            "timestamp": datetime.utcnow(),
        }

        await predictions_coll.insert_one(prediction_record)

        return PredictionResponse(
            predicted_yield=round(predicted_yield, 2),
            confidence=interpretation_data["confidence"],
            interpretation=interpretation_data["interpretation"],
            prediction_id=prediction_id,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/predictions/history")
async def get_prediction_history(
    current_user: dict = Depends(get_current_user), limit: int = 10
):
    await verify_db_connection()
    _, _, predictions_coll = await get_db_collections()

    cursor = (
        predictions_coll.find({"user_phone": current_user["phone_number"]})
        .sort("timestamp", -1)
        .limit(limit)
    )

    predictions = await cursor.to_list(length=limit)

    for prediction in predictions:
        prediction["_id"] = str(prediction["_id"])

    return {
        "predictions": predictions,
        "total": await predictions_coll.count_documents(
            {"user_phone": current_user["phone_number"]}
        ),
    }


# Data collection routes
@app.post("/api/data/submit", response_model=SubmissionResponse)
async def submit_data(
    data: DataSubmission, current_user: dict = Depends(get_current_user)
):
    await verify_db_connection()
    users_coll, submissions_coll, _ = await get_db_collections()

    try:
        submission = {
            "user_phone": current_user["phone_number"],
            "user_name": current_user["fullname"],
            "district": data.district,
            "rainfall_mm": data.rainfall_mm,
            "temperature_c": data.temperature_c,
            "soil_ph": data.soil_ph,
            "fertilizer_kg_per_ha": data.fertilizer_kg_per_ha,
            "pesticide_l_per_ha": data.pesticide_l_per_ha,
            "irrigation_type": data.irrigation_type,
            "actual_yield_tons_per_ha": data.actual_yield_tons_per_ha,
            "planting_date": data.planting_date,
            "harvest_date": data.harvest_date,
            "notes": data.notes,
            "submission_date": datetime.utcnow(),
            "status": "submitted",
        }

        result = await submissions_coll.insert_one(submission)

        await users_coll.update_one(
            {"phone_number": current_user["phone_number"]}, {"$inc": {"points": 5}}
        )

        return SubmissionResponse(
            success=True,
            message="Thank you! Your data has been submitted successfully.",
            submission_id=str(result.inserted_id),
            points_earned=5,
        )

    except Exception as e:
        return SubmissionResponse(
            success=False, message=f"Error submitting data: {str(e)}", submission_id=""
        )


@app.get("/api/data/submissions")
async def get_user_submissions(
    current_user: dict = Depends(get_current_user), limit: int = 10
):
    await verify_db_connection()
    _, submissions_coll, _ = await get_db_collections()

    cursor = (
        submissions_coll.find({"user_phone": current_user["phone_number"]})
        .sort("submission_date", -1)
        .limit(limit)
    )

    submissions = await cursor.to_list(length=limit)

    for submission in submissions:
        submission["_id"] = str(submission["_id"])

    return {"submissions": submissions}


# User profile routes
@app.get("/api/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    return {
        "fullname": current_user["fullname"],
        "phone_number": current_user["phone_number"],
        "role": current_user["role"],
        "points": current_user.get("points", 0),
        "created_at": current_user["created_at"],
        "last_login": current_user.get("last_login"),
    }


# Model management routes
@app.post("/api/model/reload")
async def reload_model(current_user: dict = Depends(get_current_user)):
    """Reload model from Google Drive (admin only)"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    global model_artifacts, model, scaler, feature_names

    try:
        model_artifacts = load_model_from_google_drive()
        if model_artifacts:
            model = model_artifacts.get("model")
            scaler = model_artifacts.get("scaler")
            feature_names = model_artifacts.get("feature_names", [])
            return {
                "message": "Model reloaded successfully",
                "features": len(feature_names),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
