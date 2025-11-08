import os
from dotenv import load_dotenv

load_dotenv()

# ===== APP CONFIG =====
APP_NAME = "Smart Gwiza - Maize Yield Prediction API"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "AI-powered maize yield prediction for Rwandan farmers"

# ===== SECURITY CONFIG =====
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-make-it-strong")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 43200  # 30 days

# ===== DATABASE CONFIG =====
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "smart_gwiza"

# ===== ML MODEL CONFIG =====
# Update this to your new model file ID
GOOGLE_DRIVE_FILE_ID = "1xXiX1zaKxM40TmbMsTZRYKgqs4nKOqrG"  # Updated file ID
GOOGLE_DRIVE_URL = (
    f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
)

# ===== CORS CONFIG =====
CORS_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# SMS Configuration
PINDO_API_TOKEN = os.getenv("PINDO_API_TOKEN")
