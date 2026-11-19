"""
SQLite bağlantı yönetimi ve veritabanı başlatma.

Hiçbir domain modülüne bağımlı değildir; yalnızca standart kütüphane kullanır.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

_SCHEMA_PATH = (
    Path(__file__).parent.parent.parent / "docs" / "database" / "schema.sql"
)

_SYSTEM_USER_ID = 1
_SYSTEM_USER_EMAIL = "system@local"


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Veritabanı bağlantısı açar, şemayı uygular ve bağlantıyı döndürür.

    İdempotent: CREATE TABLE IF NOT EXISTS kullanıldığından
    mevcut veritabanına tekrar çağrılmak güvenlidir.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # foreign_keys: bağlantı başına ayarlanmalıdır.
    conn.execute("PRAGMA foreign_keys = ON")

    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema_sql)

    # executescript implicit COMMIT yapar; foreign_keys'i yenile.
    conn.execute("PRAGMA foreign_keys = ON")

    # FR-01 tamamlanana kadar sistem kullanıcısını hazır tut.
    conn.execute(
        "INSERT OR IGNORE INTO users (id, email, password_hash)"
        " VALUES (?, ?, ?)",
        (_SYSTEM_USER_ID, _SYSTEM_USER_EMAIL, "N/A"),
    )
    conn.commit()
    return conn


@contextmanager
def get_connection(db_path: str):
    """
    Bağlam yöneticisi: hata durumunda rollback, normal çıkışta commit yapar.
    """
    conn = init_db(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
