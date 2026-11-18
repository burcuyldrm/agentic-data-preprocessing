"""
Parola hashleme (bcrypt) ve JWT token yönetimi.

PyJWT kullanılır (python-jose yerine).
python-jose[cryptography] macOS + Anaconda ortamında cryptography 43.x ile
OpenSSL çakışması nedeniyle Fatal SIGABRT veriyordu; PyJWT bu katmanı
kullanmaz ve HS256 için saf HMAC-SHA256 uygular.

bcrypt 5.x doğrudan kullanılır; passlib/bcrypt uyumluluk sorunundan kaçınmak için.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError

import src.config as config


# ---------------------------------------------------------------------------
# Parola yönetimi
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    """Düz metin parolayı bcrypt ile hashler; veritabanına saklanacak string döner."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Düz metin parola ile hash'i karşılaştırır."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT yönetimi
# ---------------------------------------------------------------------------

def create_access_token(user_id: int, email: str) -> str:
    """
    Kullanıcı bilgilerini içeren JWT access token üretir.
    Süre config.ACCESS_TOKEN_EXPIRE_MINUTES ile belirlenir.
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub":     email,
        "user_id": user_id,
        "exp":     expire,
    }
    return jwt.encode(payload, config.SECRET_KEY, algorithm=config.ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    JWT token'ı doğrular ve payload'ı döndürür.

    Raises:
        ExpiredSignatureError: Token süresi dolmuşsa (→ HTTP 403).
        InvalidTokenError:     Token geçersiz veya imzası bozuksa (→ HTTP 401).
    """
    return jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
