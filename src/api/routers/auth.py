"""
Auth router — FR-01 Kullanıcı Kaydı ve Kimlik Doğrulama.

Endpoint'ler:
  POST /auth/register  — Yeni kullanıcı oluşturur.
  POST /auth/login     — JWT access token döndürür.
  GET  /auth/me        — Kimlik doğrulaması gerektiren örnek korumalı endpoint.
"""

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, field_validator

from src.auth import create_access_token, hash_password, verify_password
from src.api.dependencies import get_current_user, get_db
from src.memory import _utc_now

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Pydantic şemaları
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Parola en az 8 karakter olmalıdır.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RegisterResponse(BaseModel):
    message: str
    user_id: int


class MeResponse(BaseModel):
    id: int
    email: str


# ---------------------------------------------------------------------------
# Endpoint'ler
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(body: RegisterRequest, conn: sqlite3.Connection = Depends(get_db)):
    """
    Yeni kullanıcı kaydı.
    Aynı e-posta ile ikinci kayıt girişiminde HTTP 409 döner.
    Parola veritabanında bcrypt hash olarak saklanır.
    """
    existing = conn.execute(
        "SELECT id FROM users WHERE email = ?",
        (body.email,),
    ).fetchone()

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Bu e-posta adresi zaten kayıtlı: {body.email}",
        )

    password_hash = hash_password(body.password)

    cursor = conn.execute(
        "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
        (body.email, password_hash, _utc_now()),
    )
    conn.commit()

    return RegisterResponse(
        message="Kullanıcı başarıyla oluşturuldu.",
        user_id=cursor.lastrowid,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, conn: sqlite3.Connection = Depends(get_db)):
    """
    Kullanıcı girişi. Doğru kimlik bilgileriyle JWT access token döner.
    Yanlış parola veya bilinmeyen e-posta → HTTP 401.
    """
    row = conn.execute(
        "SELECT id, email, password_hash FROM users WHERE email = ?",
        (body.email,),
    ).fetchone()

    if row is None or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="E-posta veya parola hatalı.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user_id=row["id"], email=row["email"])
    return TokenResponse(access_token=token)


@router.get("/me", response_model=MeResponse)
def me(current_user: dict = Depends(get_current_user)):
    """
    Token gerektiren korumalı endpoint.
    Token yoksa veya geçersizse HTTP 401, süresi dolmuşsa HTTP 403 döner.
    """
    return MeResponse(id=current_user["id"], email=current_user["email"])
