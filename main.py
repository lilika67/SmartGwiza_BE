# main.py
from fastapi import FastAPI, Depends, HTTPException, status, Body, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, field_validator, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
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
import json

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
        print(f"❌ Database connection failed: {e}")
        return False


def load_model_from_google_drive():
    """Load model directly from Google Drive"""
    try:
        print("📥 Loading model from Google Drive...")

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

        print("✅ ML Model loaded successfully from Google Drive!")
        return model_artifacts

    except Exception as e:
        print(f"❌ Error loading model from Google Drive: {e}")
        return None


def load_model_from_google_drive_with_tempfile():
    """Load model by downloading to temporary file"""
    try:
        print("📥 Downloading model from Google Drive to temporary file...")

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

        print("✅ ML Model loaded successfully via temporary file!")
        return model_artifacts

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_artifacts, model, scaler, feature_names

    print("🚀 Starting Smart Gwiza API...")

    # Initialize database connection
    db_success = await init_database()
    if not db_success:
        print("⚠️ Starting without database connection")

    # Try to load model
    model_artifacts = load_model_from_google_drive()
    if not model_artifacts:
        print("🔄 Trying alternative loading method...")
        model_artifacts = load_model_from_google_drive_with_tempfile()

    if model_artifacts:
        model = model_artifacts.get("model")
        scaler = model_artifacts.get("scaler")
        feature_names = model_artifacts.get("feature_names", [])
        print(f"✅ Model loaded successfully! Features: {len(feature_names)}")
    else:
        print("❌ Could not load ML model - predictions will not be available")

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
    district: Optional[str] = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ["farmer", "admin"]:
            raise ValueError("Role must be farmer or admin")
        return v

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, v):
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


# Admin Models
class AdminStatsResponse(BaseModel):
    total_farmers: int
    active_farmers: int
    total_predictions: int
    average_yield: float
    active_rate: float
    total_submissions: int


class FarmerListResponse(BaseModel):
    farmers: List[dict]
    total: int
    page: int
    total_pages: int


class YieldTrendResponse(BaseModel):
    trends: List[dict]


class SystemHealthResponse(BaseModel):
    database: str
    model: str
    metrics: dict
    timestamp: datetime


class BeforeAfterComparison(BaseModel):
    name: str
    before: float
    after: float


class RegionalStatsResponse(BaseModel):
    regional_stats: List[dict]


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


async def require_admin(current_user: dict = Depends(get_current_user)):
    """Dependency to require admin role"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


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
        "district": user_data.district,
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
            "user_name": current_user["fullname"],
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
        "district": current_user.get("district", ""),
        "points": current_user.get("points", 0),
        "created_at": current_user["created_at"],
        "last_login": current_user.get("last_login"),
    }


# ===== ADMIN ROUTES =====


# Admin Dashboard Statistics
@app.get("/api/admin/dashboard/stats", response_model=AdminStatsResponse)
async def get_admin_dashboard_stats(current_user: dict = Depends(require_admin)):
    """Get admin dashboard statistics"""
    # Get total farmers count
    total_farmers = await users_collection.count_documents({"role": "farmer"})

    # Get active farmers (those who logged in last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    active_farmers = await users_collection.count_documents(
        {"role": "farmer", "last_login": {"$gte": thirty_days_ago}}
    )

    # Get total predictions
    total_predictions = await predictions_collection.count_documents({})

    # Get total data submissions
    total_submissions = await submissions_collection.count_documents({})

    # Get average yield from predictions
    pipeline = [{"$group": {"_id": None, "avg_yield": {"$avg": "$predicted_yield"}}}]
    avg_yield_result = await predictions_collection.aggregate(pipeline).to_list(1)
    avg_yield = avg_yield_result[0]["avg_yield"] if avg_yield_result else 0

    # Calculate active rate
    active_rate = round(
        (active_farmers / total_farmers * 100) if total_farmers > 0 else 0, 1
    )

    return AdminStatsResponse(
        total_farmers=total_farmers,
        active_farmers=active_farmers,
        total_predictions=total_predictions,
        average_yield=round(avg_yield, 2),
        active_rate=active_rate,
        total_submissions=total_submissions,
    )


# Farmer Management
@app.get("/api/admin/farmers", response_model=FarmerListResponse)
async def get_all_farmers(
    current_user: dict = Depends(require_admin),
    page: int = 1,
    limit: int = 20,
    search: str = None,
    status: str = "all",
):
    """Get all farmers with pagination and filtering"""
    skip = (page - 1) * limit
    query = {"role": "farmer"}

    # Add search filter
    if search:
        query["$or"] = [
            {"fullname": {"$regex": search, "$options": "i"}},
            {"phone_number": {"$regex": search, "$options": "i"}},
            {"district": {"$regex": search, "$options": "i"}},
        ]

    # Add status filter
    if status == "active":
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        query["last_login"] = {"$gte": thirty_days_ago}
    elif status == "inactive":
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        query["last_login"] = {"$lt": thirty_days_ago}

    farmers_cursor = users_collection.find(query).skip(skip).limit(limit)
    total = await users_collection.count_documents(query)

    farmers = []
    async for farmer in farmers_cursor:
        # Get farmer's last prediction
        last_prediction = await predictions_collection.find_one(
            {"user_phone": farmer["phone_number"]}, sort=[("timestamp", -1)]
        )

        # Determine status based on last login
        is_active = (
            farmer.get("last_login")
            and (datetime.utcnow() - farmer["last_login"]).days <= 30
        )

        farmers.append(
            {
                "id": str(farmer["_id"]),
                "name": farmer["fullname"],
                "phone": farmer["phone_number"],
                "location": farmer.get("district", "Rwanda"),
                "last_prediction": (
                    last_prediction["timestamp"].strftime("%Y-%m-%d")
                    if last_prediction
                    else "No predictions"
                ),
                "yield": (
                    f"{last_prediction['predicted_yield']:.3f} tons/ha"
                    if last_prediction
                    else "No predictions"
                ),
                "status": "Active" if is_active else "Inactive",
                "points": farmer.get("points", 0),
                "joined_date": farmer["created_at"].strftime("%Y-%m-%d"),
            }
        )

    return FarmerListResponse(
        farmers=farmers,
        total=total,
        page=page,
        total_pages=(total + limit - 1) // limit,
    )


# Yield Trends Analytics
@app.get("/api/admin/analytics/yield-trends", response_model=YieldTrendResponse)
async def get_yield_trends(
    current_user: dict = Depends(require_admin),
    period: str = "6months",  # 1month, 3months, 6months, 1year
):
    """Get yield trends over time for analytics"""
    # Calculate date range based on period
    if period == "1month":
        days = 30
    elif period == "3months":
        days = 90
    elif period == "1year":
        days = 365
    else:  # 6months default
        days = 180

    start_date = datetime.utcnow() - timedelta(days=days)

    pipeline = [
        {"$match": {"timestamp": {"$gte": start_date}}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m", "date": "$timestamp"}},
                "average_yield": {"$avg": "$predicted_yield"},
                "prediction_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    trends = await predictions_collection.aggregate(pipeline).to_list(None)

    # Format the response
    for trend in trends:
        trend["average_yield"] = round(trend["average_yield"], 2)
        trend["month"] = trend["_id"]

    return YieldTrendResponse(trends=trends)


# Before/After Comparison
@app.get("/api/admin/analytics/before-after-comparison")
async def get_before_after_comparison(current_user: dict = Depends(require_admin)):
    """Get before/after yield comparison data"""
    # This is mock data - in a real system, you'd compare historical vs current yields
    comparison_data = [
        {"name": "Jean Baptiste", "before": 2.8, "after": 4.2},
        {"name": "Marie Claire", "before": 2.5, "after": 3.8},
        {"name": "Patrick", "before": 3.2, "after": 5.1},
        {"name": "Grace", "before": 3.0, "after": 4.5},
        {"name": "Emmanuel", "before": 2.6, "after": 3.9},
    ]

    return {"comparison": comparison_data}


# Regional Statistics
@app.get("/api/admin/analytics/regional-stats", response_model=RegionalStatsResponse)
async def get_regional_stats(current_user: dict = Depends(require_admin)):
    """Get statistics by region/district"""
    pipeline = [
        {
            "$group": {
                "_id": "$inputs.district",
                "average_yield": {"$avg": "$predicted_yield"},
                "prediction_count": {"$sum": 1},
                "farmers_count": {"$addToSet": "$user_phone"},
            }
        },
        {
            "$project": {
                "district": "$_id",
                "average_yield": {"$round": ["$average_yield", 2]},
                "prediction_count": 1,
                "farmers_count": {"$size": "$farmers_count"},
            }
        },
        {"$sort": {"average_yield": -1}},
    ]

    regional_stats = await predictions_collection.aggregate(pipeline).to_list(None)

    return RegionalStatsResponse(regional_stats=regional_stats)


# Data Export
@app.get("/api/admin/export/farmers-data")
async def export_farmers_data(
    current_user: dict = Depends(require_admin), format: str = "csv"
):
    """Export farmers data in CSV format"""
    farmers = await users_collection.find({"role": "farmer"}).to_list(None)

    if format == "csv":
        csv_data = "Farmer Name,Phone Number,District,Status,Last Prediction,Points,Joined Date\n"

        for farmer in farmers:
            last_prediction = await predictions_collection.find_one(
                {"user_phone": farmer["phone_number"]}, sort=[("timestamp", -1)]
            )

            is_active = (
                farmer.get("last_login")
                and (datetime.utcnow() - farmer["last_login"]).days <= 30
            )

            csv_data += f'"{farmer["fullname"]}",{farmer["phone_number"]},{farmer.get("district", "Rwanda")},'
            csv_data += f'{"Active" if is_active else "Inactive"},'
            csv_data += f'{last_prediction["timestamp"].strftime("%Y-%m-%d") if last_prediction else "No predictions"},'
            csv_data += f'{farmer.get("points", 0)},{farmer["created_at"].strftime("%Y-%m-%d")}\n'

        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=farmers_data_{datetime.utcnow().date()}.csv"
            },
        )


# System Health
@app.get("/api/admin/system/health", response_model=SystemHealthResponse)
async def get_system_health(current_user: dict = Depends(require_admin)):
    """Get system health and metrics"""
    # Database health
    db_status = "healthy"
    try:
        await database.command("ping")
    except Exception:
        db_status = "unhealthy"

    # Model health
    model_status = "loaded" if model is not None else "not loaded"

    # System metrics
    total_users = await users_collection.count_documents({})
    total_predictions = await predictions_collection.count_documents({})
    total_submissions = await submissions_collection.count_documents({})

    # Recent activity (last 24 hours)
    twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
    recent_predictions = await predictions_collection.count_documents(
        {"timestamp": {"$gte": twenty_four_hours_ago}}
    )
    recent_signups = await users_collection.count_documents(
        {"created_at": {"$gte": twenty_four_hours_ago}}
    )

    return SystemHealthResponse(
        database=db_status,
        model=model_status,
        metrics={
            "total_users": total_users,
            "total_predictions": total_predictions,
            "total_submissions": total_submissions,
            "recent_predictions_24h": recent_predictions,
            "recent_signups_24h": recent_signups,
        },
        timestamp=datetime.utcnow(),
    )


# Farmer Details
@app.get("/api/admin/farmers/{farmer_id}/details")
async def get_farmer_details(
    farmer_id: str, current_user: dict = Depends(require_admin)
):
    """Get detailed information about a specific farmer"""
    try:
        farmer = await users_collection.find_one({"_id": ObjectId(farmer_id)})
        if not farmer:
            raise HTTPException(status_code=404, detail="Farmer not found")

        # Get farmer's predictions
        predictions = (
            await predictions_collection.find({"user_phone": farmer["phone_number"]})
            .sort("timestamp", -1)
            .limit(10)
            .to_list(None)
        )

        # Get farmer's submissions
        submissions = (
            await submissions_collection.find({"user_phone": farmer["phone_number"]})
            .sort("submission_date", -1)
            .limit(5)
            .to_list(None)
        )

        # Format predictions and submissions
        for pred in predictions:
            pred["_id"] = str(pred["_id"])
            pred["timestamp"] = pred["timestamp"].isoformat()

        for sub in submissions:
            sub["_id"] = str(sub["_id"])
            sub["submission_date"] = sub["submission_date"].isoformat()

        return {
            "farmer_info": {
                "name": farmer["fullname"],
                "phone": farmer["phone_number"],
                "district": farmer.get("district", ""),
                "joined_date": farmer["created_at"].isoformat(),
                "last_login": farmer.get(
                    "last_login", farmer["created_at"]
                ).isoformat(),
                "points": farmer.get("points", 0),
                "status": (
                    "Active"
                    if farmer.get("last_login")
                    and (datetime.utcnow() - farmer["last_login"]).days <= 30
                    else "Inactive"
                ),
            },
            "predictions": predictions,
            "submissions": submissions,
            "prediction_count": await predictions_collection.count_documents(
                {"user_phone": farmer["phone_number"]}
            ),
            "submission_count": await submissions_collection.count_documents(
                {"user_phone": farmer["phone_number"]}
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid farmer ID: {str(e)}")


# Update Farmer Status
@app.put("/api/admin/farmers/{farmer_id}/status")
async def update_farmer_status(
    farmer_id: str,
    status: str = Body(..., embed=True),
    current_user: dict = Depends(require_admin),
):
    """Update farmer status (for future implementation)"""
    try:
        # This would update a status field in the user document
        # For now, we'll just return success
        return {
            "message": f"Farmer status would be updated to {status}",
            "success": True,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating status: {str(e)}")


# Model management routes
@app.post("/api/admin/model/reload")
async def reload_model(current_user: dict = Depends(require_admin)):
    """Reload model from Google Drive (admin only)"""
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
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading model: {str(e)}")


# Get all predictions for admin
@app.get("/api/admin/predictions")
async def get_all_predictions(
    current_user: dict = Depends(require_admin), page: int = 1, limit: int = 20
):
    """Get all predictions across all users (admin only)"""
    skip = (page - 1) * limit

    cursor = (
        predictions_collection.find({}).sort("timestamp", -1).skip(skip).limit(limit)
    )
    total = await predictions_collection.count_documents({})

    predictions = await cursor.to_list(length=limit)

    for prediction in predictions:
        prediction["_id"] = str(prediction["_id"])
        prediction["timestamp"] = prediction["timestamp"].isoformat()

    return {
        "predictions": predictions,
        "total": total,
        "page": page,
        "total_pages": (total + limit - 1) // limit,
    }


# Get all submissions for admin
@app.get("/api/admin/submissions")
async def get_all_submissions(
    current_user: dict = Depends(require_admin), page: int = 1, limit: int = 20
):
    """Get all data submissions across all users (admin only)"""
    skip = (page - 1) * limit

    cursor = (
        submissions_collection.find({})
        .sort("submission_date", -1)
        .skip(skip)
        .limit(limit)
    )
    total = await submissions_collection.count_documents({})

    submissions = await cursor.to_list(length=limit)

    for submission in submissions:
        submission["_id"] = str(submission["_id"])
        submission["submission_date"] = submission["submission_date"].isoformat()

    return {
        "submissions": submissions,
        "total": total,
        "page": page,
        "total_pages": (total + limit - 1) // limit,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
