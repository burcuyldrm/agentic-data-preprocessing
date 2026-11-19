"""
Değerlendirme denemesi kayıt işlemleri — evaluation_attempts tablosu.
"""

import json
import sqlite3

from src.core.utils import utc_now as _utc_now


def save_evaluation_attempt(
    conn: sqlite3.Connection,
    run_id: int,
    strategy_id: int,
    attempt_data: dict,
) -> int:
    """
    Değerlendirme denemesini evaluation_attempts tablosuna kaydeder.

    Başarılı veya başarısız olsun her deneme yazılır
    (FR-08 kabul kriteri 1 — tüm denemeler kaydedilir).

    attempt_data anahtarları:
      attempt_number (int, zorunlu),
      model_type, primary_metric_name,
      cv_primary_score, cv_secondary_score, cv_scores_detail (dict),
      duration_seconds, llm_call_count, status, error_message
    """
    cv_detail = attempt_data.get("cv_scores_detail")
    cv_detail_json = (
        json.dumps(cv_detail, sort_keys=True) if cv_detail else None
    )

    cursor = conn.execute(
        """
        INSERT INTO evaluation_attempts (
            analysis_run_id, strategy_id, attempt_number,
            model_type, primary_metric_name,
            cv_primary_score, cv_secondary_score, cv_scores_detail,
            duration_seconds, llm_call_count, status, error_message,
            attempted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            strategy_id,
            attempt_data["attempt_number"],
            attempt_data.get("model_type", "logistic_regression"),
            attempt_data.get("primary_metric_name", "macro_f1"),
            attempt_data.get("cv_primary_score"),
            attempt_data.get("cv_secondary_score"),
            cv_detail_json,
            attempt_data.get("duration_seconds"),
            attempt_data.get("llm_call_count", 0),
            attempt_data.get("status", "success"),
            attempt_data.get("error_message"),
            _utc_now(),
        ),
    )
    conn.commit()
    return cursor.lastrowid
