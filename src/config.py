"""
Geriye dönük uyumluluk shim — yeni kodda src.core.config kullanılması tercih edilir.
"""

from src.core.config import (  # noqa: F401
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALGORITHM,
    DB_PATH,
    SECRET_KEY,
)
