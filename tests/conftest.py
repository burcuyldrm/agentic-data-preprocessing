import os

# JWT auth testlerinden önce set edilmesi gereken env değişkenleri.
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
