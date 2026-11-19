"""
Veri seti ve profil kayıt işlemleri — datasets ve dataset_profiles tabloları.
"""

import json
import sqlite3

from src.core.utils import utc_now as _utc_now


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
