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
"""

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_PATH = (
    Path(__file__).parent.parent / "docs" / "database" / "schema.sql"
)

# Sistem kullanıcısı: FR-01 (auth) uygulanana kadar kullanılır.
_SYSTEM_USER_ID = 1
_SYSTEM_USER_EMAIL = "system@local"


# ---------------------------------------------------------------------------
# Bağlantı yönetimi
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hash yardımcıları
# ---------------------------------------------------------------------------

def compute_strategy_signature(strategy: dict) -> str:
    """
    Strateji parametrelerinden deterministik bir SHA-256 hash üretir.
    Aynı parametreler → aynı hash → preprocessing_strategies'de tekrar ekleme olmaz.
    """
    canonical = {
        "imputation_numeric":     strategy.get("imputation_numeric", "median"),
        "imputation_categorical": strategy.get("imputation_categorical", "most_frequent"),
        "scaling_method":         strategy.get("scaling_method", "standard"),
        "encoding_method":        strategy.get("encoding_method", "onehot"),
        "outlier_handling":       strategy.get("outlier_handling", "none"),
        "feature_selection":      strategy.get("feature_selection", "none"),
    }
    extra = strategy.get("extra_params")
    if extra:
        canonical["extra_params"] = json.dumps(extra, sort_keys=True)

    payload = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_profile_hash(profile_summary: dict) -> str:
    """
    Veri profili özet istatistiklerinden benzerlik hash'i üretir.

    Kesin eşleşme yerine istatistikleri demetlere (bucket) böler;
    benzer veri setleri aynı hash'e düşer ve deneyim belleğindeki
    geçmiş stratejiler bunlara uygulanabilir.

    Demet boyutları:
      num_bucket   — numeric sütun adedi // 3
      cat_bucket   — categorical sütun adedi // 2
      miss_bucket  — toplam eksik değer % // 10
    """
    num_cols     = profile_summary.get("numeric_column_count", 0)
    cat_cols     = profile_summary.get("categorical_column_count", 0)
    missing_pct  = profile_summary.get("total_missing_pct", 0.0)
    problem_type = profile_summary.get("problem_type", "classification")

    num_bucket  = num_cols // 3
    cat_bucket  = cat_cols // 2
    miss_bucket = int(missing_pct // 10)

    signature = (
        f"{problem_type}|"
        f"num:{num_bucket}|"
        f"cat:{cat_bucket}|"
        f"miss:{miss_bucket}"
    )
    return hashlib.sha256(signature.encode()).hexdigest()


def build_profile_summary(
    profile: dict,
    problem_type: str,
) -> dict:
    """
    generate_data_profile() çıktısını compute_profile_hash() girdisine dönüştürür.

    Args:
        profile:      src.data_profile.generate_data_profile() dönüş değeri.
        problem_type: 'classification' veya 'regression'.

    Returns:
        compute_profile_hash() ve save_dataset_profile() için ortak özet sözlüğü.
    """
    column_types = profile.get("column_types", {})
    missing = profile.get("missing_values", {})

    num_cols = len(column_types.get("numeric", []))
    cat_cols = len(column_types.get("categorical", []))

    pct_values = [
        v.get("missing_percent", 0.0)
        for v in missing.values()
        if isinstance(v, dict)
    ]
    total_missing_pct = sum(pct_values) / len(pct_values) if pct_values else 0.0

    return {
        "numeric_column_count":     num_cols,
        "categorical_column_count": cat_cols,
        "total_missing_pct":        round(total_missing_pct, 4),
        "problem_type":             problem_type,
    }


# ---------------------------------------------------------------------------
# Strateji kataloğu
# ---------------------------------------------------------------------------

def upsert_strategy(conn: sqlite3.Connection, strategy: dict) -> int:
    """
    preprocessing_strategies tablosunda stratejiyi bulur ya da oluşturur.

    strategy_signature çakışması → mevcut satırın id'si döner (UNIQUE kısıt gereği).

    Args:
        strategy: imputation_numeric, imputation_categorical, scaling_method,
                  encoding_method, [outlier_handling], [feature_selection],
                  [extra_params] anahtarlarını içeren sözlük.

    Returns:
        strategy_id (int)
    """
    signature = compute_strategy_signature(strategy)

    row = conn.execute(
        "SELECT id FROM preprocessing_strategies WHERE strategy_signature = ?",
        (signature,),
    ).fetchone()
    if row:
        return row["id"]

    extra = strategy.get("extra_params")
    extra_json = json.dumps(extra, sort_keys=True) if extra else None

    cursor = conn.execute(
        """
        INSERT INTO preprocessing_strategies (
            imputation_numeric, imputation_categorical, scaling_method,
            encoding_method, outlier_handling, feature_selection,
            extra_params, strategy_signature, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            strategy.get("imputation_numeric", "median"),
            strategy.get("imputation_categorical", "most_frequent"),
            strategy.get("scaling_method", "standard"),
            strategy.get("encoding_method", "onehot"),
            strategy.get("outlier_handling", "none"),
            strategy.get("feature_selection", "none"),
            extra_json,
            signature,
            _utc_now(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Veri seti ve profil kaydı
# ---------------------------------------------------------------------------

def save_dataset(conn: sqlite3.Connection, dataset_info: dict) -> int:
    """
    Veri setini datasets tablosuna kaydeder.

    file_hash UNIQUE kısıtı sayesinde aynı dosya farklı adla tekrar kaydedilmez
    (FR-08 kabul kriteri 5 — hash ile tekrar kullanımı engelleme).

    dataset_info anahtarları:
      name, file_path, file_hash (SHA-256), [source], [row_count],
      [column_count], [file_size_bytes], [source_dataset_id], [download_status]
    """
    file_hash = dataset_info["file_hash"]

    row = conn.execute(
        "SELECT id FROM datasets WHERE file_hash = ?",
        (file_hash,),
    ).fetchone()
    if row:
        return row["id"]

    cursor = conn.execute(
        """
        INSERT INTO datasets (
            source, source_dataset_id, name, file_path,
            file_hash, row_count, column_count, file_size_bytes,
            download_status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_info.get("source", "upload"),
            dataset_info.get("source_dataset_id"),
            dataset_info["name"],
            dataset_info["file_path"],
            file_hash,
            dataset_info.get("row_count"),
            dataset_info.get("column_count"),
            dataset_info.get("file_size_bytes"),
            dataset_info.get("download_status", "ready"),
            _utc_now(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def save_dataset_profile(
    conn: sqlite3.Connection,
    dataset_id: int,
    profile: dict,
    profile_summary: dict,
    profile_hash: str,
) -> int:
    """
    Profil istatistiklerini dataset_profiles tablosuna kaydeder.
    dataset_id için kayıt zaten varsa mevcut id'yi döndürür (UNIQUE kısıt).

    Args:
        profile:         generate_data_profile() çıktısı.
        profile_summary: build_profile_summary() çıktısı.
        profile_hash:    compute_profile_hash() çıktısı.
    """
    row = conn.execute(
        "SELECT id FROM dataset_profiles WHERE dataset_id = ?",
        (dataset_id,),
    ).fetchone()
    if row:
        return row["id"]

    dataset_info = profile.get("dataset_info", {})
    missing      = profile.get("missing_values", {})
    distribution = profile.get("distribution", {})

    missing_detail = {
        col: v.get("missing_percent", 0.0)
        for col, v in missing.items()
        if isinstance(v, dict)
    }
    skewness_detail = {
        col: v.get("skewness", 0.0)
        for col, v in distribution.items()
        if isinstance(v, dict)
    }
    numeric_stats = {
        col: {
            "mean":     v.get("mean"),
            "std":      v.get("std"),
            "skewness": v.get("skewness"),
        }
        for col, v in distribution.items()
        if isinstance(v, dict)
    }

    cursor = conn.execute(
        """
        INSERT INTO dataset_profiles (
            dataset_id,
            numeric_column_count, categorical_column_count,
            total_missing_pct, missing_value_detail,
            target_column, problem_type,
            train_row_count, test_row_count,
            numeric_stats, skewness_detail,
            profile_hash, profiled_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            dataset_id,
            profile_summary["numeric_column_count"],
            profile_summary["categorical_column_count"],
            profile_summary["total_missing_pct"],
            json.dumps(missing_detail, sort_keys=True),
            dataset_info.get("target", ""),
            profile_summary["problem_type"],
            profile.get("train_row_count"),
            profile.get("test_row_count"),
            json.dumps(numeric_stats, sort_keys=True),
            json.dumps(skewness_detail, sort_keys=True),
            profile_hash,
            _utc_now(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


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
# Değerlendirme denemesi kaydı
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deneyim belleği — UPSERT
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deneyim belleği — Sorgulama
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Hata toleranslı sarmalayıcı
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Özel yardımcılar
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
