from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
import uuid
import asyncio
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


# SMS Service Class (Integrated directly in farmer.py)
class PindoSMSService:
    def __init__(self):
        import os

        self.api_token = os.getenv("PINDO_API_TOKEN")
        self.base_url = "https://api.pindo.io/v1/sms/"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    async def send_sms(self, to_number: str, message: str, sender_name: str = "Pindo"):
        """Send single SMS to a user"""
        if not self.api_token:
            print(" SMS service not configured - PINDO_API_TOKEN missing")
            return {"success": False, "error": "SMS service not configured"}

        # Validate phone number format (Rwanda)
        if not to_number.startswith("+250"):
            to_number = "+250" + to_number.lstrip("0")

        # Use approved sender IDs for Rwanda
        # Common approved sender IDs: "Pindo", short codes, or alphanumeric (max 11 chars)
        approved_sender = "PindoTest"

        data = {"to": to_number, "text": message, "sender": approved_sender}

        try:
            import requests

            print(
                f"📱 Attempting to send SMS to {to_number} with sender: {approved_sender}"
            )
            response = requests.post(self.base_url, json=data, headers=self.headers)
            result = response.json()

            if response.status_code == 201:
                print(f" SMS sent successfully to {to_number}")
                return {
                    "success": True,
                    "sms_id": result.get("sms_id"),
                    "remaining_balance": result.get("remaining_balance"),
                    "total_cost": result.get("total_cost"),
                    "status": "sent",
                }
            else:
                print(f" SMS failed with status {response.status_code}: {result}")
                return {
                    "success": False,
                    "error": result,
                    "status_code": response.status_code,
                }

        except Exception as e:
            print(f" SMS error: {str(e)}")
            return {"success": False, "error": str(e)}

    async def send_prediction_notification(
        self, user_phone: str, user_name: str, prediction_data: dict, input_data: dict
    ):
        """Send prediction results via SMS with all input details"""
        predicted_yield = prediction_data["predicted_yield"]
        interpretation = prediction_data["interpretation"]
        prediction_id = prediction_data["prediction_id"]

        # Create Version 1 message only
        message = f"Muraho neza {user_name}! Mwakoze gukoresha SmartGwiza !\n\n"
        message += "Ibipimo mwakoresheje mugereranya ni ibi:\n"
        message += f"Imvura: {input_data.get('rainfall_mm', 0)}mm \n"
        message += f"Ubushyuhe: {input_data.get('temperature_c', 0)}°C\n"
        message += f"pH y'ubutaka: {input_data.get('soil_ph', 0)}\n "
        message += f"Ifumbire: {input_data.get('fertilizer_used_kg_per_ha', 0)}kg\n"
        message += f"Umuti wica ibyonnyi: {input_data.get('pesticide_l_per_ha', 0)}L\n "
        message += f"ubwoko bwo kuhira: {input_data.get('irrigation_type', 'N/A')}\n"
        message += f"Umusaruro uteganyijwe {predicted_yield}t/ha \n"
        

        return await self.send_sms(user_phone, message)


# Create global SMS service instance
sms_service = PindoSMSService()


# ===== PREDICTION ROUTES =====
@router.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    input_data: PredictionInput, current_user: dict = Depends(get_current_user)
):
    """
    Make crop yield prediction and send results via SMS
    """
    await verify_db_connection()
    _, _, predictions_coll = await get_db_collections()

    try:
        # Make prediction
        predicted_yield = predict_yield(input_data)
        interpretation_data = get_interpretation(predicted_yield)
        prediction_id = str(uuid.uuid4())

        # Create prediction record
        prediction_record = {
            "prediction_id": prediction_id,
            "user_phone": current_user["phone_number"],
            "user_name": current_user["fullname"],
            "inputs": input_data.dict(),
            "predicted_yield": predicted_yield,
            "interpretation": interpretation_data,
            "timestamp": datetime.utcnow(),
            "sms_sent": False,  # We'll update this after SMS attempt
        }

        # Save to database first
        predictions_coll.insert_one(prediction_record)

        # Prepare SMS data
        sms_data = {
            "predicted_yield": round(predicted_yield, 2),
            "interpretation": interpretation_data["interpretation"],
            "prediction_id": prediction_id,
        }

        # Send SMS notification in background (non-blocking)
        asyncio.create_task(
            send_prediction_sms_with_retry(
                current_user["phone_number"],
                current_user["fullname"],
                sms_data,
                input_data.dict(),  # Pass ALL input data
                prediction_id,
                predictions_coll,
            )
        )

        # Return API response immediately
        return PredictionResponse(
            predicted_yield=round(predicted_yield, 2),
            confidence=interpretation_data["confidence"],
            interpretation=interpretation_data["interpretation"],
            prediction_id=prediction_id,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


async def send_prediction_sms_with_retry(
    phone: str,
    name: str,
    prediction_data: dict,
    input_data: dict,
    prediction_id: str,
    predictions_coll,
):
    """Send SMS and update database with result"""
    try:
        result = await sms_service.send_prediction_notification(
            phone, name, prediction_data, input_data
        )

        # Update the prediction record with SMS status
        predictions_coll.update_one(
            {"prediction_id": prediction_id},
            {
                "$set": {
                    "sms_sent": result.get("success", False),
                    "sms_result": result,
                    "sms_timestamp": datetime.utcnow(),
                }
            },
        )

        if result.get("success"):
            print(f" Prediction SMS sent successfully for {prediction_id}")
            print(
                f"SMS included: Rainfall={input_data.get('rainfall_mm')}mm, "
                f"Temperature={input_data.get('temperature_c')}°C, "
                f"pH={input_data.get('soil_ph')}, "
                f"Fertilizer={input_data.get('fertilizer_used_kg_per_ha')}kg/ha, "
                f"Pesticide={input_data.get('pesticide_l_per_ha')}L/ha, "
                f"Irrigation={input_data.get('irrigation_type')}"
            )
        else:
            print(f" Failed to send SMS for {prediction_id}: {result.get('error')}")

    except Exception as e:
        print(f" Error in SMS background task: {str(e)}")


@router.get("/predictions/history")
async def get_prediction_history(
    current_user: dict = Depends(get_current_user), limit: int = 10
):
    """
    Get user's prediction history
    """
    await verify_db_connection()
    _, _, predictions_coll = await get_db_collections()

    cursor = (
        predictions_coll.find({"user_phone": current_user["phone_number"]})
        .sort("timestamp", -1)
        .limit(limit)
    )

    predictions = list(cursor)

    # Convert ObjectId to string for JSON serialization
    for prediction in predictions:
        prediction["_id"] = str(prediction["_id"])
        # Convert datetime to ISO format if needed
        if isinstance(prediction.get("timestamp"), datetime):
            prediction["timestamp"] = prediction["timestamp"].isoformat()

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
    """
    Submit farming data (NO SMS sent for data submission)
    """
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
            "yield_before": data.yield_before,
            "actual_yield_tons_per_ha": data.actual_yield_tons_per_ha,
            "planting_date": data.planting_date,
            "harvest_date": data.harvest_date,
            "notes": data.notes,
            "submission_date": datetime.utcnow(),
            "status": "submitted",
        }

        result = submissions_coll.insert_one(submission)

        # Update user points (no SMS for this)
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
            success=False,
            message=f"Error submitting data: {str(e)}",
            submission_id="",
            points_earned=0,
        )


@router.get("/data/submissions")
async def get_user_submissions(
    current_user: dict = Depends(get_current_user), limit: int = 10
):
    """
    Get user's data submissions (NO SMS involved)
    """
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
        # Convert datetime to ISO format if needed
        if isinstance(submission.get("submission_date"), datetime):
            submission["submission_date"] = submission["submission_date"].isoformat()

    return {"submissions": submissions}


# ===== USER PROFILE ROUTES =====
@router.get("/user/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """
    
    """
    return {
        "fullname": current_user["fullname"],
        "phone_number": current_user["phone_number"],
        "role": current_user["role"],
        "district": current_user.get("district", ""),
        "points": current_user.get("points", 0),
        "created_at": current_user["created_at"],
        "last_login": current_user.get("last_login"),
    }


# ===== SMS TEST ENDPOINT (Optional - for testing) =====
@router.post("/test-sms")
async def test_sms_notification(current_user: dict = Depends(get_current_user)):
    """
    Test endpoint to verify SMS functionality with all inputs
    """
    test_prediction_data = {
        "predicted_yield": 8.5,
        "interpretation": "Good yield expected with current conditions",
        "prediction_id": "test_" + str(uuid.uuid4())[:8],
    }

    # Create test input data with all fields
    test_input_data = {
        "rainfall_mm": 1200,
        "temperature_c": 25,
        "soil_ph": 6.5,
        "fertilizer_used_kg_per_ha": 150,
        "pesticide_l_per_ha": 2,
        "irrigation_type": "Drip",
    }

    result = await sms_service.send_prediction_notification(
        current_user["phone_number"],
        current_user["fullname"],
        test_prediction_data,
        test_input_data,
    )

    return {
        "message": "Test SMS sent with all input data",
        "sms_result": result,
        "user_phone": current_user["phone_number"],
        "inputs_included": list(test_input_data.keys()),
    }


# ===== SMS STATUS ENDPOINT =====
@router.get("/predictions/{prediction_id}/sms-status")
async def get_sms_status(
    prediction_id: str, current_user: dict = Depends(get_current_user)
):
    """
    Check SMS delivery status for a specific prediction
    """
    await verify_db_connection()
    _, _, predictions_coll = await get_db_collections()

    prediction = predictions_coll.find_one(
        {"prediction_id": prediction_id, "user_phone": current_user["phone_number"]}
    )

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return {
        "prediction_id": prediction_id,
        "sms_sent": prediction.get("sms_sent", False),
        "sms_result": prediction.get("sms_result"),
        "sms_timestamp": prediction.get("sms_timestamp"),
        "inputs_used": prediction.get("inputs", {}),
    }
