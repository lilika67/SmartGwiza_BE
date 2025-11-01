from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import uuid
from auth import get_current_user
from models import (
    PredictionInput,
    PredictionResponse,
    DataSubmission,
    SubmissionResponse,
)
from database import get_db_collections, verify_db_connection
from ml_model import predict_yield, get_interpretation

router = APIRouter(prefix="/api", tags=["farmer"])


# ===== PREDICTION ROUTES =====
@router.post("/predict", response_model=PredictionResponse)
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

        predictions_coll.insert_one(prediction_record)

        return PredictionResponse(
            predicted_yield=round(predicted_yield, 2),
            confidence=interpretation_data["confidence"],
            interpretation=interpretation_data["interpretation"],
            prediction_id=prediction_id,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/predictions/history")
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

    predictions = list(cursor)
    for prediction in predictions:
        prediction["_id"] = str(prediction["_id"])

    return {
        "predictions": predictions,
        "total": predictions_coll.count_documents(
            {"user_phone": current_user["phone_number"]}
        ),
    }


# ===== DATA SUBMISSION ROUTES =====
@router.post("/data/submit", response_model=SubmissionResponse)
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

        result = submissions_coll.insert_one(submission)

        users_coll.update_one(
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


@router.get("/data/submissions")
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

    submissions = list(cursor)
    for submission in submissions:
        submission["_id"] = str(submission["_id"])

    return {"submissions": submissions}


# ===== USER PROFILE ROUTES =====
@router.get("/user/profile")
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
