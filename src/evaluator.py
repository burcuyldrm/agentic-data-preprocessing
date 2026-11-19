"""
Geriye dönük uyumluluk shim — yeni kodda src.evaluation paketi kullanılması tercih edilir.
"""

from src.evaluation.metrics import (  # noqa: F401
    VALID_ENCODING_METHOD,
    VALID_FEATURE_SELECTION,
    VALID_IMPUTATION_CATEGORICAL,
    VALID_IMPUTATION_NUMERIC,
    VALID_OUTLIER_HANDLING,
    VALID_SCALING_METHOD,
    build_preprocessor_from_strategy,
    evaluate_on_test,
    evaluate_strategy,
    validate_strategy,
)
