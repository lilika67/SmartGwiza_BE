from fastapi import HTTPException
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()

# Global variables
client = None
db = None
users_collection = None
submissions_collection = None
predictions_collection = None


# ===== DATABASE FUNCTIONS =====
async def init_database():
    """Initialize database connection"""
    global client, db, users_collection, submissions_collection, predictions_collection
    try:
        client = MongoClient(
            os.getenv("MONGODB_URL", "mongodb://localhost:27017"),
            server_api=ServerApi("1"),
            tls=True,
            tlsAllowInvalidCertificates=True,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=30000,
            maxPoolSize=50,
            retryWrites=True,
            w="majority",
        )

        client.admin.command("ping", serverSelectionTimeoutMS=5000)
        db = client.smart_gwiza
        users_collection = db.users
        submissions_collection = db.submissions
        predictions_collection = db.predictions

        print(" Database connected successfully")
        return True
    except Exception as e:
        print(f" Database connection failed: {str(e)}")
        if client:
            client.close()
        client = None
        db = None
        users_collection = None
        submissions_collection = None
        predictions_collection = None
        return False


async def get_db_collections():
    """Get database collections with safety check"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return users_collection, submissions_collection, predictions_collection


async def verify_db_connection():
    """Verify database is available"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")



async def get_db_collections():
    """Get database collections with safety check"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return users_collection, submissions_collection, predictions_collection


async def verify_db_connection():
    """Verify database is available"""
    if db is None:
        raise HTTPException(status_code=503, detail="Database service unavailable")
