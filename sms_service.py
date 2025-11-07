import os
import requests
from typing import Dict, List, Optional
from fastapi import HTTPException


class PindoSMSService:
    def __init__(self):
        self.api_token = os.getenv("PINDO_API_TOKEN")
        self.base_url = "https://api.pindo.io/v1/sms/"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

    async def send_sms(
        self, to_number: str, message: str, sender_name: str = "SmartGwiza"
    ) -> Dict:
        """Send single SMS to a user"""
        if not self.api_token:
            raise HTTPException(status_code=500, detail="SMS service not configured")

        data = {"to": to_number, "text": message, "sender": sender_name}

        try:
            response = requests.post(self.base_url, json=data, headers=self.headers)
            result = response.json()

            if response.status_code == 201:
                return {
                    "success": True,
                    "sms_id": result.get("sms_id"),
                    "remaining_balance": result.get("remaining_balance"),
                    "total_cost": result.get("total_cost"),
                    "status": "sent",
                }
            else:
                return {
                    "success": False,
                    "error": result,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_prediction_notification(
        self, user_phone: str, user_name: str, prediction_data: Dict
    ):
        """Send prediction results via SMS"""
        predicted_yield = prediction_data["predicted_yield"]
        interpretation = prediction_data["interpretation"]
        prediction_id = prediction_data["prediction_id"]

        message = f"Hello {user_name}! \n"
        message += f"Your crop yield prediction: {predicted_yield} tons/ha\n"
        message += f"Interpretation: {interpretation}\n"
        message += f"Prediction ID: {prediction_id[:8]}...\n"
        message += "Thank you for using Smart Gwiza!"

        return await self.send_sms(user_phone, message)

    async def send_data_submission_confirmation(
        self, user_phone: str, user_name: str, submission_id: str, points_earned: int
    ):
        """Send confirmation for data submission"""
        message = f"Hello {user_name}! \n"
        message += "Your farming data has been submitted successfully!\n"
        message += f"Earned: {points_earned} points\n"
        message += f"Submission ID: {submission_id[:8]}...\n"
        message += "Thank you for contributing to Smart Gwiza!"

        return await self.send_sms(user_phone, message)

    async def send_welcome_message(self, user_phone: str, user_name: str):
        """Send welcome message to new users"""
        message = f"Welcome {user_name}! \n"
        message += "Thank you for joining Smart Gwiza!\n"
        message += "Get crop predictions & share data to earn points.\n"
        message += "Start by submitting your farm data!"

        return await self.send_sms(user_phone, message)


# Create global instance
sms_service = PindoSMSService()
