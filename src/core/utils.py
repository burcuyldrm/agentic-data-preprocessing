"""
Uygulama genelinde kullanılan yardımcı fonksiyonlar.
"""

from datetime import datetime, timezone


def utc_now() -> str:
    """UTC şimdiki zamanı SQLite-uyumlu string olarak döndürür."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
