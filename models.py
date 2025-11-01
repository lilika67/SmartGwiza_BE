from pydantic import BaseModel, field_validator, Field
from datetime import datetime
from typing import Optional, List


# Authentication Models
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


# Prediction Models
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


# Data Submission Models
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


class RegionalStatsResponse(BaseModel):
    regional_stats: List[dict]


# Add this to your models.py file
class SignupResponse(BaseModel):
    success: bool
    message: str
    role: str
    fullname: str
    user_id: str
