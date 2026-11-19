"""
Geriye dönük uyumluluk re-export'ları.
Yeni kodda doğrudan alt modüller kullanılması tercih edilir:
  src.evaluation.metrics     — validate_strategy, build_preprocessor_from_strategy,
                               evaluate_strategy, evaluate_on_test, VALID_* sabitleri
  src.evaluation.repository  — save_evaluation_attempt
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
from src.evaluation.repository import save_evaluation_attempt  # noqa: F401
