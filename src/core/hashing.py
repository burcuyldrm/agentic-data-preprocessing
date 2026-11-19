"""
Deterministik SHA-256 hash yardımcıları.

compute_strategy_signature — Strateji parametrelerinden tekrarlanabilir hash.
compute_profile_hash       — Veri profil özetinden benzerlik hash'i.
"""

import hashlib
import json


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
