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


def _run_migrations(conn: sqlite3.Connection) -> None:
    """
    Mevcut veritabanına AI-03 sütunlarını ekler (backward compat).
    PRAGMA table_info ile varlık kontrolü yapar; yoksa ALTER TABLE uygular.
    """
    migrations = [
        ("analysis_runs",    "max_llm_calls",        "INTEGER NOT NULL DEFAULT 15"),
        ("analysis_reports", "total_provider_calls", "INTEGER NOT NULL DEFAULT 0"),
        ("analysis_reports", "termination_reason",   "TEXT"),
    ]
    for table, column, definition in migrations:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    conn.commit()


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

    # Mevcut veritabanlarına AI-03 sütunlarını ekle.
    _run_migrations(conn)

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
