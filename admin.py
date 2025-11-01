from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
from typing import List, Optional
from auth import get_current_user, require_admin
from database import get_db_collections, verify_db_connection
from models import (
    AdminStatsResponse,
    FarmerListResponse,
    YieldTrendResponse,
    SystemHealthResponse,
    RegionalStatsResponse,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


# ===== ADMIN ROUTES =====
@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(current_user: dict = Depends(require_admin)):
    """Get overall system statistics"""
    await verify_db_connection()
    users_coll, submissions_coll, predictions_coll = await get_db_collections()

    try:
        # Total farmers
        total_farmers = users_coll.count_documents({"role": "farmer"})

        # Active farmers (logged in last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        active_farmers = users_coll.count_documents(
            {"role": "farmer", "last_login": {"$gte": thirty_days_ago}}
        )

        # Total predictions
        total_predictions = predictions_coll.count_documents({})

        # Total submissions
        total_submissions = submissions_coll.count_documents({})

        # Average yield from submissions
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_yield": {"$avg": "$actual_yield_tons_per_ha"},
                }
            }
        ]
        avg_yield_result = list(submissions_coll.aggregate(pipeline))
        average_yield = avg_yield_result[0]["avg_yield"] if avg_yield_result else 0

        # Active rate
        active_rate = (active_farmers / total_farmers * 100) if total_farmers > 0 else 0

        return AdminStatsResponse(
            total_farmers=total_farmers,
            active_farmers=active_farmers,
            total_predictions=total_predictions,
            average_yield=round(average_yield, 2),
            active_rate=round(active_rate, 2),
            total_submissions=total_submissions,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@router.get("/farmers", response_model=FarmerListResponse)
async def get_farmers_list(
    current_user: dict = Depends(require_admin),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
):
    """Get paginated list of farmers"""
    await verify_db_connection()
    users_coll, submissions_coll, predictions_coll = await get_db_collections()

    try:
        skip = (page - 1) * limit

        # Define thirty_days_ago here
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        # Get farmers with pagination
        farmers_cursor = (
            users_coll.find({"role": "farmer"})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )

        farmers = list(farmers_cursor)
        total_farmers = users_coll.count_documents({"role": "farmer"})

        # Enrich farmer data with additional stats
        enriched_farmers = []
        for farmer in farmers:
            phone = farmer["phone_number"]

            # Count predictions and submissions
            prediction_count = predictions_coll.count_documents({"user_phone": phone})
            submission_count = submissions_coll.count_documents({"user_phone": phone})

            # Get latest submission if exists
            latest_submission = submissions_coll.find_one(
                {"user_phone": phone}, sort=[("submission_date", -1)]
            )

            enriched_farmer = {
                "fullname": farmer["fullname"],
                "phone_number": farmer["phone_number"],
                "district": farmer.get("district", ""),
                "points": farmer.get("points", 0),
                "created_at": farmer["created_at"],
                "last_login": farmer.get("last_login"),
                "prediction_count": prediction_count,
                "submission_count": submission_count,
                "latest_yield": (
                    latest_submission.get("actual_yield_tons_per_ha")
                    if latest_submission
                    else None
                ),
                "is_active": farmer.get("last_login", datetime.min) >= thirty_days_ago,
            }
            enriched_farmers.append(enriched_farmer)

        return FarmerListResponse(
            farmers=enriched_farmers,
            total=total_farmers,
            page=page,
            total_pages=(total_farmers + limit - 1) // limit,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching farmers: {str(e)}")


@router.get("/yield-trends", response_model=YieldTrendResponse)
async def get_yield_trends(
    current_user: dict = Depends(require_admin), days: int = Query(30, ge=1, le=365)
):
    """Get yield trends over time"""
    await verify_db_connection()
    _, submissions_coll, _ = await get_db_collections()

    try:
        start_date = datetime.utcnow() - timedelta(days=days)

        pipeline = [
            {"$match": {"submission_date": {"$gte": start_date}}},
            {
                "$group": {
                    "_id": {
                        "year": {"$year": "$submission_date"},
                        "month": {"$month": "$submission_date"},
                        "day": {"$dayOfMonth": "$submission_date"},
                    },
                    "average_yield": {"$avg": "$actual_yield_tons_per_ha"},
                    "submission_count": {"$sum": 1},
                    "min_yield": {"$min": "$actual_yield_tons_per_ha"},
                    "max_yield": {"$max": "$actual_yield_tons_per_ha"},
                }
            },
            {"$sort": {"_id.year": 1, "_id.month": 1, "_id.day": 1}},
        ]

        trends = list(submissions_coll.aggregate(pipeline))

        # Format the response
        formatted_trends = []
        for trend in trends:
            date_str = f"{trend['_id']['year']}-{trend['_id']['month']:02d}-{trend['_id']['day']:02d}"
            formatted_trends.append(
                {
                    "date": date_str,
                    "average_yield": round(trend["average_yield"], 2),
                    "submission_count": trend["submission_count"],
                    "min_yield": trend["min_yield"],
                    "max_yield": trend["max_yield"],
                }
            )

        return YieldTrendResponse(trends=formatted_trends)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching yield trends: {str(e)}"
        )


@router.get("/regional-stats", response_model=RegionalStatsResponse)
async def get_regional_stats(current_user: dict = Depends(require_admin)):
    """Get statistics by region/district"""
    await verify_db_connection()
    users_coll, submissions_coll, predictions_coll = await get_db_collections()

    try:
        # Regional stats from submissions
        pipeline = [
            {
                "$group": {
                    "_id": "$district",
                    "average_yield": {"$avg": "$actual_yield_tons_per_ha"},
                    "total_submissions": {"$sum": 1},
                    "farmers_count": {"$addToSet": "$user_phone"},
                }
            },
            {
                "$project": {
                    "district": "$_id",
                    "average_yield": {"$round": ["$average_yield", 2]},
                    "total_submissions": 1,
                    "farmers_count": {"$size": "$farmers_count"},
                }
            },
            {"$sort": {"average_yield": -1}},
        ]

        regional_stats = list(submissions_coll.aggregate(pipeline))

        return RegionalStatsResponse(regional_stats=regional_stats)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching regional stats: {str(e)}"
        )


@router.get("/system-health", response_model=SystemHealthResponse)
async def get_system_health(current_user: dict = Depends(require_admin)):
    """Get system health information"""
    await verify_db_connection()
    users_coll, submissions_coll, predictions_coll = await get_db_collections()

    try:
        # Database status
        from database import db

        db_status = "connected" if db is not None else "disconnected"

        # Model status
        from ml_model import model

        model_status = "loaded" if model is not None else "not loaded"

        # System metrics
        total_users = users_coll.count_documents({})
        total_predictions = predictions_coll.count_documents({})
        total_submissions = submissions_coll.count_documents({})

        # Recent activity (last 24 hours)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        recent_predictions = predictions_coll.count_documents(
            {"timestamp": {"$gte": twenty_four_hours_ago}}
        )
        recent_submissions = submissions_coll.count_documents(
            {"submission_date": {"$gte": twenty_four_hours_ago}}
        )

        metrics = {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "total_submissions": total_submissions,
            "recent_predictions_24h": recent_predictions,
            "recent_submissions_24h": recent_submissions,
            "uptime": "running",  # In production, you might calculate actual uptime
        }

        return SystemHealthResponse(
            database=db_status,
            model=model_status,
            metrics=metrics,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching system health: {str(e)}"
        )


@router.get("/submissions/recent")
async def get_recent_submissions(
    current_user: dict = Depends(require_admin), limit: int = Query(20, ge=1, le=100)
):
    """Get recent data submissions"""
    await verify_db_connection()
    _, submissions_coll, _ = await get_db_collections()

    try:
        cursor = submissions_coll.find().sort("submission_date", -1).limit(limit)
        submissions = list(cursor)

        for submission in submissions:
            submission["_id"] = str(submission["_id"])

        return {"submissions": submissions}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching recent submissions: {str(e)}"
        )


@router.get("/predictions/recent")
async def get_recent_predictions(
    current_user: dict = Depends(require_admin), limit: int = Query(20, ge=1, le=100)
):
    """Get recent predictions"""
    await verify_db_connection()
    _, _, predictions_coll = await get_db_collections()

    try:
        cursor = predictions_coll.find().sort("timestamp", -1).limit(limit)
        predictions = list(cursor)

        for prediction in predictions:
            prediction["_id"] = str(prediction["_id"])

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching recent predictions: {str(e)}"
        )
