from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from auth import router as auth_router
from farmer import router as farmer_router
from database import init_database
from ml_model import initialize_model
from config import *
# Add this import at the top of main.py
from admin import router as admin_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Smart Gwiza API...")

    db_success = await init_database()
    if not db_success:
        print("Starting without database connection")

    model_success = initialize_model()
    if not model_success:
        print(" Could not load ML model - predictions will not be available")

    yield
    # Shutdown
    from database import client

    if client:
        client.close()


# ===== FASTAPI APP =====
app = FastAPI(
    title=APP_NAME,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# Include routers
app.include_router(auth_router)
app.include_router(farmer_router)
app.include_router(admin_router)

@app.get("/")
async def root():
    from database import db
    from ml_model import model

    return {
        "message": "Smart Gwiza API",
        "version": APP_VERSION,
        "status": "running",
        "model_loaded": model is not None,
        "database_connected": db is not None,
    }


@app.get("/health")
async def health_check():
    from datetime import datetime
    from database import db
    from ml_model import model

    db_status = "disconnected"
    if db is not None:
        try:
            db.client.admin.command("ping")
            db_status = "connected"
        except Exception:
            db_status = "disconnected"

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": model is not None,
        "database": db_status,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
