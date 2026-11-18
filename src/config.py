"""
Ortam değişkenlerinden uygulama ayarlarını yükler.
Değerler .env dosyasından okunur; dosya yoksa os ortam değişkenleri kullanılır.
"""

import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH: str = os.getenv("DB_PATH", "agentic_preprocessing.db")

# Üretimde mutlaka güçlü rastgele bir değerle değiştirilmeli.
SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGE_ME_before_production")

ALGORITHM: str = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)
