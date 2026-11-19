"""
SQLite tabanlı deneyim belleği — FR-08.

Dışa açık fonksiyonlar:
  init_db                    — Veritabanını başlatır (şema + sistem kullanıcısı).
  compute_strategy_signature — Strateji parametrelerinden deterministik hash üretir.
  compute_profile_hash       — Benzerlik araması için profil hash'i üretir.
  build_profile_summary      — generate_data_profile çıktısını hash girdisine dönüştürür.
  upsert_strategy            — Stratejiyi bulur ya da oluşturur; strategy_id döndürür.
  save_dataset               — Veri setini kaydeder; SHA-256 ile tekrar yüklemeyi engeller.
  save_dataset_profile       — Profil istatistiklerini kaydeder.
  create_analysis_run        — Yeni analiz çalıştırması oluşturur; run_id döndürür.
  update_analysis_run        — Analiz durumunu ve sayacını günceller.
  save_evaluation_attempt    — Deneme sonucunu (başarılı veya başarısız) kaydeder.
  update_experience_memory   — Belleği UPSERT ile günceller.
  query_similar_strategies   — Benzer profilden başarılı stratejileri döndürür.
  safe_record_attempt        — Hata durumunu ana akışa yansıtmayan sarmalayıcı.
  save_analysis_report       — Analiz raporunu kaydeder.
"""

import json
import logging
import sqlite3
from typing import Optional

from src.core.hashing import compute_profile_hash, compute_strategy_signature  # noqa: F401
from src.core.utils import utc_now as _utc_now
from src.datasets.repository import (  # noqa: F401
    build_profile_summary,
    save_dataset,
    save_dataset_profile,
)
from src.evaluation.repository import save_evaluation_attempt  # noqa: F401
from src.experience_memory.store import (  # noqa: F401
    query_similar_strategies,
    safe_record_attempt,
    update_experience_memory,
)
from src.persistence.db import get_connection, init_db  # noqa: F401
from src.preprocessing.repository import upsert_strategy  # noqa: F401

logger = logging.getLogger(__name__)

# FR-01 tamamlanana kadar sistem kullanıcısı varsayılan değeri için gereklidir.
_SYSTEM_USER_ID = 1


# ---------------------------------------------------------------------------
# Analiz çalıştırması yönetimi
# ---------------------------------------------------------------------------

def create_analysis_run(
    conn: sqlite3.Connection,
    dataset_id: int,
    user_request: str,
    experiment_mode: str = "D",
    random_seed: int = 42,
    max_attempts: int = 5,
    user_id: int = _SYSTEM_USER_ID,
) -> int:
    """
    Yeni bir analiz çalıştırması oluşturur; run_id döndürür.
    FR-01 tamamlanana kadar user_id varsayılan olarak sistem kullanıcısını kullanır.
    """
    cursor = conn.execute(
        """
        INSERT INTO analysis_runs (
            user_id, dataset_id, user_request,
            status, experiment_mode, random_seed,
            max_attempts, attempt_count, started_at
        ) VALUES (?, ?, ?, 'running', ?, ?, ?, 0, ?)
        """,
        (
            user_id, dataset_id, user_request,
            experiment_mode, random_seed, max_attempts,
            _utc_now(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_analysis_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    attempt_count: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Analiz çalıştırması durumunu günceller.
    status 'completed', 'failed' veya 'cancelled' ise finished_at doldurulur.
    """
    finished = (
        _utc_now()
        if status in ("completed", "failed", "cancelled")
        else None
    )

    if attempt_count is not None:
        conn.execute(
            """
            UPDATE analysis_runs
            SET status        = ?,
                attempt_count = ?,
                finished_at   = ?,
                error_message = ?
            WHERE id = ?
            """,
            (status, attempt_count, finished, error_message, run_id),
        )
    else:
        conn.execute(
            """
            UPDATE analysis_runs
            SET status        = ?,
                finished_at   = ?,
                error_message = ?
            WHERE id = ?
            """,
            (status, finished, error_message, run_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Analiz raporu
# ---------------------------------------------------------------------------

def save_analysis_report(
    conn: sqlite3.Connection,
    run_id: int,
    best_strategy_id: Optional[int],
    final_scores: dict,
    cv_primary_score: Optional[float] = None,
    total_attempts: int = 0,
    total_duration_seconds: Optional[float] = None,
    total_llm_calls: int = 0,
    selection_rationale: Optional[str] = None,
) -> int:
    """
    Analiz çalıştırmasının nihai raporunu analysis_reports tablosuna kaydeder.

    Her analysis_run_id için yalnızca bir rapor olabilir (UNIQUE kısıt).
    final_scores, evaluate_on_test() çıktısıyla uyumlu anahtarlar içermelidir.
    """
    metrics_detail = final_scores.get("final_test_metrics_detail")
    metrics_json   = json.dumps(metrics_detail, sort_keys=True) if metrics_detail else None

    cursor = conn.execute(
        """
        INSERT INTO analysis_reports (
            analysis_run_id, best_strategy_id,
            final_test_primary_score, final_test_secondary_score,
            final_test_metrics_detail, cv_primary_score,
            total_attempts, total_duration_seconds, total_llm_calls,
            selection_rationale, report_created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            best_strategy_id,
            final_scores.get("final_test_primary_score"),
            final_scores.get("final_test_secondary_score"),
            metrics_json,
            cv_primary_score,
            total_attempts,
            total_duration_seconds,
            total_llm_calls,
            selection_rationale,
            _utc_now(),
        ),
    )
    conn.commit()
    return cursor.lastrowid
