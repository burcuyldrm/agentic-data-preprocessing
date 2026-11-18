"""
FastAPI uygulama giriş noktası.

Çalıştırma:
  uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI

from src.api.routers import auth as auth_router

app = FastAPI(
    title="Agentic Data Preprocessing API",
    description=(
        "Çok ajanlı otonom veri analizi ve ön işleme sistemi — "
        "Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip"
    ),
    version="0.1.0",
)

app.include_router(auth_router.router)
