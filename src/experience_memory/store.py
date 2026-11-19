"""
Deneyim belleği işlemleri — experience_memory tablosu.

update_experience_memory  — Belleği UPSERT ile günceller.
query_similar_strategies  — Benzer profilden başarılı stratejileri döndürür.
safe_record_attempt       — Değerlendirme kaydını hata toleranslı biçimde yürütür.
"""

import logging
import sqlite3
from typing import Optional

from src.core.utils import utc_now as _utc_now
from src.evaluation.repository import save_evaluation_attempt

logger = logging.getLogger(__name__)


def update_experience_memory(
    conn: sqlite3.Connection,
    profile_hash: str,
    strategy_id: int,
    problem_type: str,
    cv_score: float,
    duration: Optional[float],
    is_success: bool,
    dataset_id: Optional[int] = None,
    run_id: Optional[int] = None,
) -> None:
    """
    (profile_hash, strategy_id, problem_type) üçlüsüne karşılık gelen
    experience_memory satırını UPSERT ile günceller.

    - Kayıt yoksa: yeni satır oluşturur.
    - Kayıt varsa: hareketli ortalama skoru, en iyi skoru,
      deneme sayısını ve başarı sayısını günceller.
    """
    now = _utc_now()

    existing = conn.execute(
        """
        SELECT id, cv_score_avg, cv_score_best,
               attempt_count, success_count, avg_duration_seconds
        FROM experience_memory
        WHERE profile_hash = ? AND strategy_id = ? AND problem_type = ?
        """,
        (profile_hash, strategy_id, problem_type),
    ).fetchone()

    if existing is None:
        conn.execute(
            """
            INSERT INTO experience_memory (
                profile_hash, strategy_id, problem_type,
                cv_score_avg, cv_score_best,
                attempt_count, success_count,
                avg_duration_seconds,
                last_dataset_id, last_analysis_run_id,
                last_updated_at
            ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                profile_hash,
                strategy_id,
                problem_type,
                cv_score,
                cv_score,
                1 if is_success else 0,
                duration,
                dataset_id,
                run_id,
                now,
            ),
        )
    else:
        old_n   = existing["attempt_count"]
        new_n   = old_n + 1
        new_avg = (existing["cv_score_avg"] * old_n + cv_score) / new_n
        new_best = max(existing["cv_score_best"], cv_score)
        new_success = existing["success_count"] + (1 if is_success else 0)

        old_dur = existing["avg_duration_seconds"]
        if duration is not None and old_dur is not None:
            new_avg_dur: Optional[float] = (old_dur * old_n + duration) / new_n
        elif duration is not None:
            new_avg_dur = duration
        else:
            new_avg_dur = old_dur

        conn.execute(
            """
            UPDATE experience_memory
            SET cv_score_avg         = ?,
                cv_score_best        = ?,
                attempt_count        = ?,
                success_count        = ?,
                avg_duration_seconds = ?,
                last_dataset_id      = ?,
                last_analysis_run_id = ?,
                last_updated_at      = ?
            WHERE profile_hash = ? AND strategy_id = ? AND problem_type = ?
            """,
            (
                new_avg, new_best, new_n, new_success,
                new_avg_dur, dataset_id, run_id, now,
                profile_hash, strategy_id, problem_type,
            ),
        )
    conn.commit()


def query_similar_strategies(
    conn: sqlite3.Connection,
    profile_hash: str,
    problem_type: str,
    exclude_dataset_id: Optional[int] = None,
    limit: int = 5,
) -> list:
    """
    profile_hash ve problem_type eşleşmesine göre experience_memory'deki
    geçmiş stratejileri döndürür.

    exclude_dataset_id ile mevcut veri setine ait kayıtlar sorgudan çıkarılır.
    Bu mekanizma, bir veri setinin kendi önceki performansından faydalanmasını
    önleyerek veri sızıntısını engeller (FR-08 kabul kriteri 4).

    Sonuçlar cv_score_avg DESC, success_rate DESC sırasıyla gelir.
    """
    if exclude_dataset_id is not None:
        rows = conn.execute(
            """
            SELECT
                vsp.memory_id,
                vsp.strategy_id,
                vsp.strategy_signature,
                vsp.imputation_numeric,
                vsp.imputation_categorical,
                vsp.scaling_method,
                vsp.encoding_method,
                vsp.outlier_handling,
                vsp.feature_selection,
                vsp.cv_score_avg,
                vsp.cv_score_best,
                vsp.attempt_count,
                vsp.success_count,
                vsp.success_rate,
                vsp.avg_duration_seconds
            FROM v_strategy_performance vsp
            WHERE vsp.profile_hash = ?
              AND vsp.problem_type = ?
              AND vsp.memory_id NOT IN (
                      SELECT id
                      FROM experience_memory
                      WHERE last_dataset_id = ?
                  )
            ORDER BY vsp.cv_score_avg DESC, vsp.success_rate DESC
            LIMIT ?
            """,
            (profile_hash, problem_type, exclude_dataset_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT
                vsp.memory_id,
                vsp.strategy_id,
                vsp.strategy_signature,
                vsp.imputation_numeric,
                vsp.imputation_categorical,
                vsp.scaling_method,
                vsp.encoding_method,
                vsp.outlier_handling,
                vsp.feature_selection,
                vsp.cv_score_avg,
                vsp.cv_score_best,
                vsp.attempt_count,
                vsp.success_count,
                vsp.success_rate,
                vsp.avg_duration_seconds
            FROM v_strategy_performance vsp
            WHERE vsp.profile_hash = ?
              AND vsp.problem_type = ?
            ORDER BY vsp.cv_score_avg DESC, vsp.success_rate DESC
            LIMIT ?
            """,
            (profile_hash, problem_type, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def safe_record_attempt(
    conn: sqlite3.Connection,
    run_id: int,
    strategy_id: int,
    profile_hash: str,
    problem_type: str,
    attempt_data: dict,
    dataset_id: Optional[int] = None,
) -> Optional[int]:
    """
    save_evaluation_attempt ve update_experience_memory çağrılarını
    hata toleranslı biçimde yürütür.

    Bellek kaydı başarısız olursa hata loglanır ama exception fırlatılmaz;
    ana analiz akışı kesintisiz devam eder (FR-08 kabul kriteri 6).

    Başarı durumunda evaluation_attempts.id döner, hata durumunda None.
    """
    try:
        attempt_id = save_evaluation_attempt(conn, run_id, strategy_id, attempt_data)

        cv_score   = attempt_data.get("cv_primary_score") or 0.0
        duration   = attempt_data.get("duration_seconds")
        is_success = attempt_data.get("status") == "success"

        update_experience_memory(
            conn,
            profile_hash,
            strategy_id,
            problem_type,
            cv_score,
            duration,
            is_success,
            dataset_id,
            run_id,
        )
        return attempt_id

    except Exception as exc:
        logger.error(
            "Bellek kaydı başarısız (run_id=%s, strategy_id=%s): %s",
            run_id,
            strategy_id,
            exc,
            exc_info=True,
        )
        return None
