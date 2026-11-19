"""
Geriye dönük uyumluluk re-export'ları.
Yeni kodda doğrudan alt modüller kullanılması tercih edilir:
  src.preprocessing.pipeline    — build_baseline_preprocessor, AdaptivePreprocessor
  src.preprocessing.strategy    — select_preprocessing_strategy
  src.preprocessing.repository  — upsert_strategy
"""

from src.preprocessing.pipeline import AdaptivePreprocessor, build_baseline_preprocessor  # noqa: F401
from src.preprocessing.repository import upsert_strategy  # noqa: F401
from src.preprocessing.strategy import select_preprocessing_strategy  # noqa: F401
