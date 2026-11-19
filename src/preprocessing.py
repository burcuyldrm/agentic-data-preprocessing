"""
Geriye dönük uyumluluk shim — yeni kodda src.preprocessing paketi kullanılması tercih edilir.
"""

from src.preprocessing.pipeline import AdaptivePreprocessor, build_baseline_preprocessor  # noqa: F401
