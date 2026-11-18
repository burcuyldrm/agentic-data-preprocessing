"""
FastAPI bağımlılıkları (Depends):
  get_db           — Her istek için SQLite bağlantısı açar ve kapatır.
  get_current_user — Bearer token'ı doğrular; kullanıcı dict'i döndürür.
"""

import sqlite3
from typing import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt import ExpiredSignatureError, InvalidTokenError

import src.config as config
from src.auth import decode_access_token
from src.memory import init_db

_security = HTTPBearer()


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    İstek başına tek bir SQLite bağlantısı açar.
    İstek tamamlandığında (hata olsa bile) bağlantı kapatılır.
    """
    conn = init_db(config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    """
    Bearer token'dan kullanıcıyı çıkarır.

    - Token yoksa veya geçersizse       → HTTP 401
    - Token süresi dolmuşsa             → HTTP 403
    - Kullanıcı veritabanında yoksa     → HTTP 401
    """
    token = credentials.credentials

    try:
        payload = decode_access_token(token)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token süresi dolmuş.",
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email: str = payload.get("sub", "")
    user_id: int = payload.get("user_id", 0)

    row = conn.execute(
        "SELECT id, email FROM users WHERE id = ? AND email = ?",
        (user_id, email),
    ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Kullanıcı bulunamadı.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"id": row["id"], "email": row["email"]}
