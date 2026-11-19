"""
Ön işleme stratejisi katalog işlemleri — preprocessing_strategies tablosu.
"""

import json
import sqlite3

from src.core.hashing import compute_strategy_signature
from src.core.utils import utc_now as _utc_now


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
