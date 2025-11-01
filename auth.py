from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import re
import hashlib
from pydantic import BaseModel  # Add this import
from models import UserSignup, UserLogin, Token
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from database import get_db_collections, verify_db_connection

router = APIRouter(prefix="/api/auth", tags=["authentication"])
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ===== AUTH UTILITIES =====
def safe_hash_password(password: str) -> str:
    """Safely hash password, handling bcrypt's 72-byte limit"""
    password_bytes = password.encode("utf-8")
    if len(password_bytes) <= 72:
        return pwd_context.hash(password)
    pre_hashed = hashlib.sha256(password_bytes).hexdigest()
    return pwd_context.hash(pre_hashed)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    if pwd_context.verify(plain_password, hashed_password):
        return True
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
        status_code=401,
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

    await verify_db_connection()
    users_coll, _, _ = await get_db_collections()
    user = users_coll.find_one({"phone_number": phone_number})
    if user is None:
        raise credentials_exception
    return user


async def require_admin(current_user: dict = Depends(get_current_user)):
    """Dependency to require admin role"""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ===== RESPONSE MODELS =====
class SignupResponse(BaseModel):
    success: bool
    message: str
    access_token: str
    token_type: str
    role: str
    fullname: str
    user_id: str


# ===== AUTH ROUTES =====
@router.post("/signup", response_model=SignupResponse)
async def signup(user_data: UserSignup):
    await verify_db_connection()
    users_coll, _, _ = await get_db_collections()

    normalized_phone = validate_rwandan_phone(user_data.phone_number)

    existing_user = users_coll.find_one({"phone_number": normalized_phone})
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

    result = users_coll.insert_one(user_doc)
    user_id = str(result.inserted_id)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": normalized_phone, "role": user_data.role},
        expires_delta=access_token_expires,
    )

    return SignupResponse(
        success=True,
        message="Account created successfully! Welcome to Smart Gwiza.",
        access_token=access_token,
        token_type="bearer",
        role=user_data.role,
        fullname=user_data.fullname,
        user_id=user_id,
    )


@router.post("/login", response_model=Token)
async def login(user_data: UserLogin):
    await verify_db_connection()
    users_coll, _, _ = await get_db_collections()

    normalized_phone = validate_rwandan_phone(user_data.phone_number)

    user = users_coll.find_one({"phone_number": normalized_phone})
    if not user or not verify_password(user_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid phone number or password")

    users_coll.update_one(
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
