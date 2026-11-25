"""
FR-07 — LLM strateji öneri arayüzü.

OpenAI-uyumlu sağlayıcı adaptörü ve ortam değişkeni yapılandırması.
Varsayılan çalışma modu stub kalır (LLM_PROVIDER ayarlanmadıkça).
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geçerli strateji değerleri — src/evaluation/metrics.py ile senkron
# ---------------------------------------------------------------------------

VALID_IMPUTATION_NUMERIC     = {"mean", "median", "knn"}
VALID_IMPUTATION_CATEGORICAL = {"most_frequent", "constant"}
VALID_SCALING_METHOD         = {"standard", "robust", "minmax", "none"}
VALID_ENCODING_METHOD        = {"onehot"}
VALID_OUTLIER_HANDLING       = {"clip", "remove", "none"}
VALID_FEATURE_SELECTION      = {"none", "variance_threshold", "selectkbest"}

_REQUIRED_STRATEGY_FIELDS = frozenset({
    "imputation_numeric",
    "imputation_categorical",
    "scaling_method",
    "encoding_method",
    "outlier_handling",
    "feature_selection",
})

_FIELD_VALIDATORS: dict[str, set] = {
    "imputation_numeric":     VALID_IMPUTATION_NUMERIC,
    "imputation_categorical": VALID_IMPUTATION_CATEGORICAL,
    "scaling_method":         VALID_SCALING_METHOD,
    "encoding_method":        VALID_ENCODING_METHOD,
    "outlier_handling":       VALID_OUTLIER_HANDLING,
    "feature_selection":      VALID_FEATURE_SELECTION,
}


# ---------------------------------------------------------------------------
# İstisna
# ---------------------------------------------------------------------------

class LLMConfigurationError(Exception):
    """LLM istemcisi yapılandırma hatası."""


# ---------------------------------------------------------------------------
# Soyut arayüz
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Yapılandırma
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """LLM sağlayıcı yapılandırması."""
    provider:         str   = "stub"
    model:            str   = "gpt-4o-mini"
    api_key:          str   = ""
    base_url:         str   = ""
    timeout_seconds:  float = 30.0
    max_retries:      int   = 2
    fallback_enabled: bool  = False

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.environ.get("LLM_PROVIDER", "stub").lower().strip()
        model    = os.environ.get("LLM_MODEL", "gpt-4o-mini").strip()
        api_key  = os.environ.get("LLM_API_KEY", "").strip()
        base_url = os.environ.get("LLM_BASE_URL", "").strip()

        raw_timeout = os.environ.get("LLM_TIMEOUT_SECONDS", "30.0")
        try:
            timeout_seconds = float(raw_timeout)
            if timeout_seconds <= 0:
                raise ValueError("non-positive")
        except (ValueError, TypeError):
            raise LLMConfigurationError(
                f"LLM_TIMEOUT_SECONDS geçersiz: {raw_timeout!r}. Pozitif sayı bekleniyor."
            )

        raw_retries = os.environ.get("LLM_MAX_RETRIES", "2")
        try:
            max_retries = int(raw_retries)
            if max_retries < 0:
                raise ValueError("negative")
        except (ValueError, TypeError):
            raise LLMConfigurationError(
                f"LLM_MAX_RETRIES geçersiz: {raw_retries!r}. Negatif olmayan tam sayı bekleniyor."
            )

        raw_fallback     = os.environ.get("LLM_FALLBACK_ENABLED", "false").lower().strip()
        fallback_enabled = raw_fallback in ("1", "true", "yes")

        if provider == "openai" and not api_key:
            raise LLMConfigurationError(
                "LLM_PROVIDER=openai seçildi ancak LLM_API_KEY ayarlanmadı."
            )

        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            fallback_enabled=fallback_enabled,
        )


# ---------------------------------------------------------------------------
# Prompt yardımcıları
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a machine learning preprocessing expert.\n"
    "Given a dataset profile summary and previously tried strategies, "
    "suggest a new preprocessing strategy.\n"
    "Reply with ONLY valid JSON matching this exact schema:\n"
    "{\n"
    '  "imputation_numeric":     "mean" | "median" | "knn",\n'
    '  "imputation_categorical": "most_frequent" | "constant",\n'
    '  "scaling_method":         "standard" | "robust" | "minmax" | "none",\n'
    '  "encoding_method":        "onehot",\n'
    '  "outlier_handling":       "clip" | "remove" | "none",\n'
    '  "feature_selection":      "none" | "variance_threshold" | "selectkbest"\n'
    "}\n"
    "No explanation. No markdown. No extra keys. JSON only."
)


def _build_prompt(
    profile_summary: dict,
    task_type: str,
    tried_signatures: set[str],
    attempt_number: int,
) -> str:
    """Profile stats sadece — ham veri yok."""
    tried_list = sorted(tried_signatures)
    tried_str  = "\n".join(f"  - {s}" for s in tried_list) if tried_list else "  (none)"
    return (
        f"Dataset profile:\n"
        f"  numeric_column_count: {profile_summary.get('numeric_column_count', 0)}\n"
        f"  categorical_column_count: {profile_summary.get('categorical_column_count', 0)}\n"
        f"  total_missing_pct: {profile_summary.get('total_missing_pct', 0.0):.2f}\n"
        f"  problem_type: {task_type}\n\n"
        f"Attempt number: {attempt_number}\n\n"
        f"Previously tried strategy signatures (do NOT repeat):\n{tried_str}\n\n"
        f"Suggest a new preprocessing strategy different from all above."
    )


def _parse_and_validate_strategy(raw: str) -> Optional[dict]:
    """JSON ayrıştır, alanları doğrula, bilinmeyen anahtarları çıkar."""
    if not raw or not raw.strip():
        logger.warning("LLM boş yanıt döndürdü.")
        return None
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        logger.warning("LLM yanıtı JSON ayrıştırılamadı: %s", exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("LLM yanıtı dict değil: %s", type(parsed).__name__)
        return None
    missing = _REQUIRED_STRATEGY_FIELDS - parsed.keys()
    if missing:
        logger.warning("LLM yanıtında eksik alanlar: %s", sorted(missing))
        return None
    unknown = parsed.keys() - _REQUIRED_STRATEGY_FIELDS
    if unknown:
        logger.warning("LLM yanıtında bilinmeyen anahtarlar: %s", sorted(unknown))
        return None
    for field_name, valid_values in _FIELD_VALIDATORS.items():
        val = parsed.get(field_name)
        if val not in valid_values:
            logger.warning(
                "LLM yanıtında geçersiz değer: %s=%r (geçerliler: %s)",
                field_name, val, valid_values,
            )
            return None
    return {k: parsed[k] for k in _REQUIRED_STRATEGY_FIELDS}


def _get_retry_after(exc: Exception) -> Optional[float]:
    """Hata nesnesinden Retry-After süresini (saniye) çıkar."""
    for attr in ("retry_after", "headers"):
        val = getattr(exc, attr, None)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            return float(val)
        if hasattr(val, "get"):
            header = val.get("Retry-After") or val.get("retry-after")
            if header:
                try:
                    return float(header)
                except (ValueError, TypeError):
                    pass
    return None


# ---------------------------------------------------------------------------
# OpenAI istemcisi
# ---------------------------------------------------------------------------

class OpenAILLMClient(LLMClient):
    """
    OpenAI-uyumlu sağlayıcı adaptörü.

    - Üstel geri çekilmeli yeniden deneme (zaman aşımı, hız limiti, sunucu hatası)
    - Kimlik doğrulama hataları yeniden denenMEZ
    - Fallback: tükenince StubLLMClient'e düşme (isteğe bağlı)
    - API anahtarı günlüklere yazılmaz
    """

    def __init__(self, config: LLMConfig) -> None:
        try:
            import openai as _openai
        except ImportError as exc:
            raise LLMConfigurationError(
                "openai paketi bulunamadı. `pip install openai` çalıştırın."
            ) from exc

        self._config      = config
        self._openai      = _openai
        self._fallback    = StubLLMClient() if config.fallback_enabled else None
        self._stub_active = False

        client_kwargs: dict = {
            "api_key": config.api_key,
            "timeout": config.timeout_seconds,
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self._client = _openai.OpenAI(**client_kwargs)

    @property
    def fallback_activated(self) -> bool:
        """True ise stub fallback'e geçilmiştir."""
        return self._stub_active

    def suggest_strategy(
        self,
        profile_summary: dict,
        task_type: str,
        tried_signatures: set[str],
        attempt_number: int,
    ) -> Optional[dict]:
        if self._stub_active and self._fallback is not None:
            return self._fallback.suggest_strategy(
                profile_summary, task_type, tried_signatures, attempt_number
            )

        prompt    = _build_prompt(profile_summary, task_type, tried_signatures, attempt_number)
        last_exc: Optional[Exception] = None
        delay     = 1.0

        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                raw    = response.choices[0].message.content or ""
                result = _parse_and_validate_strategy(raw)
                if result is not None:
                    return result
                logger.warning("LLM geçersiz strateji döndürdü (deneme %d).", attempt + 1)
                return None  # geçersiz yanıt yeniden denemeyi hak etmez

            except self._openai.AuthenticationError as exc:
                logger.error("LLM kimlik doğrulama hatası. API anahtarını kontrol edin.")
                raise LLMConfigurationError(
                    "LLM API anahtarı geçersiz veya eksik."
                ) from exc

            except (
                self._openai.APITimeoutError,
                self._openai.RateLimitError,
                self._openai.InternalServerError,
                self._openai.APIConnectionError,
            ) as exc:
                last_exc    = exc
                retry_after = _get_retry_after(exc)
                sleep_secs  = min(retry_after if retry_after else delay, 10.0)
                logger.warning(
                    "LLM geçici hata (deneme %d/%d): %s. %.1fs beklenecek.",
                    attempt + 1,
                    self._config.max_retries + 1,
                    type(exc).__name__,
                    sleep_secs,
                )
                if attempt < self._config.max_retries:
                    time.sleep(sleep_secs)
                    delay = min(delay * 2, 10.0)

            except Exception as exc:
                logger.error("LLM beklenmeyen hata: %s", type(exc).__name__)
                last_exc = exc
                break

        logger.warning(
            "LLM %d denemeden sonra yanıt vermedi: %s",
            self._config.max_retries + 1,
            type(last_exc).__name__ if last_exc else "bilinmeyen hata",
        )
        if self._fallback is not None:
            logger.info("Fallback stub istemciye geçiliyor.")
            self._stub_active = True
            return self._fallback.suggest_strategy(
                profile_summary, task_type, tried_signatures, attempt_number
            )
        return None


# ---------------------------------------------------------------------------
# Fabrika
# ---------------------------------------------------------------------------

def create_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """
    Ortam değişkenlerine göre uygun LLM istemcisini oluşturur.

    LLM_PROVIDER ayarlanmamışsa veya 'stub' ise StubLLMClient döner.
    """
    if config is None:
        config = LLMConfig.from_env()

    if config.provider in ("stub", ""):
        return StubLLMClient()

    if config.provider == "openai":
        return OpenAILLMClient(config)

    raise LLMConfigurationError(
        f"Bilinmeyen LLM sağlayıcısı: {config.provider!r}. "
        "Desteklenen: 'stub', 'openai'."
    )
