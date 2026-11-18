"""
FR-07 — LLM strateji öneri arayüzü.

Gerçek LLM entegrasyonu sonraki aşamaya ertelenmiştir.
StubLLMClient, testlerde ve yerel geliştirmede kullanılan
deterministik bir alternatif sağlar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """LLM arka ucundan bağımsız strateji öneri arayüzü."""

    @abstractmethod
    def suggest_strategy(
        self,
        profile_summary: dict,
        task_type: str,
        tried_signatures: set[str],
        attempt_number: int,
    ) -> Optional[dict]:
        """
        Veri profili ve daha önce denenen stratejiler baz alınarak
        yeni bir önişleme stratejisi önerir.

        Args:
            profile_summary:  build_profile_summary() çıktısı.
            task_type:        'classification' veya 'regression'.
            tried_signatures: Daha önce denenen strategy_signature kümesi.
            attempt_number:   1-tabanlı mevcut deneme numarası.

        Returns:
            Strateji parametreleri dict'i veya None (öneri tükenirse).
        """


# ---------------------------------------------------------------------------
# Deterministik stub — gerçek LLM YOK
# ---------------------------------------------------------------------------

# 6 sabit aday; sırayla önerilir.
_CANDIDATES: list[dict] = [
    {
        "imputation_numeric":     "median",
        "imputation_categorical": "most_frequent",
        "scaling_method":         "standard",
        "encoding_method":        "onehot",
        "outlier_handling":       "none",
        "feature_selection":      "none",
    },
    {
        "imputation_numeric":     "median",
        "imputation_categorical": "most_frequent",
        "scaling_method":         "robust",
        "encoding_method":        "onehot",
        "outlier_handling":       "none",
        "feature_selection":      "none",
    },
    {
        "imputation_numeric":     "mean",
        "imputation_categorical": "most_frequent",
        "scaling_method":         "standard",
        "encoding_method":        "onehot",
        "outlier_handling":       "none",
        "feature_selection":      "none",
    },
    {
        "imputation_numeric":     "median",
        "imputation_categorical": "most_frequent",
        "scaling_method":         "minmax",
        "encoding_method":        "onehot",
        "outlier_handling":       "none",
        "feature_selection":      "none",
    },
    {
        "imputation_numeric":     "knn",
        "imputation_categorical": "most_frequent",
        "scaling_method":         "robust",
        "encoding_method":        "onehot",
        "outlier_handling":       "none",
        "feature_selection":      "none",
    },
    {
        "imputation_numeric":     "median",
        "imputation_categorical": "most_frequent",
        "scaling_method":         "none",
        "encoding_method":        "onehot",
        "outlier_handling":       "none",
        "feature_selection":      "none",
    },
]


class StubLLMClient(LLMClient):
    """
    Deterministik, sıralı strateji önericisi.
    Bu bir gerçek LLM değildir; testler ve geliştirme ortamı için vardır.
    6 sabit adayı denenmemiş ilk sırayla önerir; hepsi denenince None döner.
    """

    def suggest_strategy(
        self,
        profile_summary: dict,
        task_type: str,
        tried_signatures: set[str],
        attempt_number: int,
    ) -> Optional[dict]:
        from src.memory import compute_strategy_signature

        for candidate in _CANDIDATES:
            if compute_strategy_signature(candidate) not in tried_signatures:
                return dict(candidate)
        return None
