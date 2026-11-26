"""
LLM istemcisi birim testleri — gerçek API çağrısı yok.

LLMConfig.from_env, _parse_and_validate_strategy, _build_prompt,
create_llm_client, OpenAILLMClient retry/fallback davranışları.
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.llm_client import (
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    OpenAILLMClient,
    StubLLMClient,
    _build_prompt,
    _parse_and_validate_strategy,
    create_llm_client,
)


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

_VALID_STRATEGY_JSON = json.dumps({
    "imputation_numeric":     "median",
    "imputation_categorical": "most_frequent",
    "scaling_method":         "standard",
    "encoding_method":        "onehot",
    "outlier_handling":       "none",
    "feature_selection":      "none",
})

_PROFILE = {
    "numeric_column_count":     5,
    "categorical_column_count": 2,
    "total_missing_pct":        0.12,
}


def _make_exc_classes():
    """Mock openai modülü için ayrı exception sınıfları."""
    AuthenticationError = type("AuthenticationError", (Exception,), {})
    APITimeoutError     = type("APITimeoutError",     (Exception,), {})
    RateLimitError      = type("RateLimitError",      (Exception,), {})
    InternalServerError = type("InternalServerError", (Exception,), {})
    APIConnectionError  = type("APIConnectionError",  (Exception,), {})
    return AuthenticationError, APITimeoutError, RateLimitError, InternalServerError, APIConnectionError


def _make_mock_openai(response_content: str = _VALID_STRATEGY_JSON):
    (
        AuthenticationError, APITimeoutError, RateLimitError,
        InternalServerError, APIConnectionError,
    ) = _make_exc_classes()

    mock_message         = MagicMock()
    mock_message.content = response_content
    mock_choice          = MagicMock()
    mock_choice.message  = mock_message
    mock_response        = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    mock_openai                     = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    mock_openai.AuthenticationError = AuthenticationError
    mock_openai.APITimeoutError     = APITimeoutError
    mock_openai.RateLimitError      = RateLimitError
    mock_openai.InternalServerError = InternalServerError
    mock_openai.APIConnectionError  = APIConnectionError

    return mock_openai, mock_client, mock_response


def _make_openai_client(
    cfg: LLMConfig = LLMConfig(provider="openai", api_key="sk-test", max_retries=0),
    response_content: str = _VALID_STRATEGY_JSON,
) -> OpenAILLMClient:
    """__init__ çağırmadan OpenAILLMClient oluşturur."""
    mock_openai, mock_underlying, _ = _make_mock_openai(response_content)
    client           = object.__new__(OpenAILLMClient)
    client._config   = cfg
    client._openai   = mock_openai
    client._client   = mock_underlying
    client._fallback = None
    client._tls      = threading.local()
    return client


# ---------------------------------------------------------------------------
# LLMConfig.from_env
# ---------------------------------------------------------------------------

class TestLLMConfigFromEnv:
    def test_default_is_stub(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        config = LLMConfig.from_env()
        assert config.provider == "stub"

    def test_openai_selection(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        config = LLMConfig.from_env()
        assert config.provider == "openai"
        assert config.api_key == "sk-test"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
            LLMConfig.from_env()

    def test_invalid_timeout_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "not-a-number")
        with pytest.raises(LLMConfigurationError, match="LLM_TIMEOUT_SECONDS"):
            LLMConfig.from_env()

    def test_zero_timeout_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0")
        with pytest.raises(LLMConfigurationError, match="LLM_TIMEOUT_SECONDS"):
            LLMConfig.from_env()

    def test_negative_timeout_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "-5.0")
        with pytest.raises(LLMConfigurationError, match="LLM_TIMEOUT_SECONDS"):
            LLMConfig.from_env()

    def test_invalid_retries_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_MAX_RETRIES", "not-an-int")
        with pytest.raises(LLMConfigurationError, match="LLM_MAX_RETRIES"):
            LLMConfig.from_env()

    def test_negative_retries_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_MAX_RETRIES", "-1")
        with pytest.raises(LLMConfigurationError, match="LLM_MAX_RETRIES"):
            LLMConfig.from_env()

    def test_fallback_true_values(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        for val in ("1", "true", "yes"):
            monkeypatch.setenv("LLM_FALLBACK_ENABLED", val)
            config = LLMConfig.from_env()
            assert config.fallback_enabled is True, f"{val!r} için başarısız"

    def test_fallback_false_by_default(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
        config = LLMConfig.from_env()
        assert config.fallback_enabled is False

    def test_base_url_passthrough(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_BASE_URL", "https://my-proxy.example.com/v1")
        config = LLMConfig.from_env()
        assert config.base_url == "https://my-proxy.example.com/v1"

    def test_model_passthrough(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        config = LLMConfig.from_env()
        assert config.model == "gpt-4o"

    def test_timeout_passthrough(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60.0")
        config = LLMConfig.from_env()
        assert config.timeout_seconds == 60.0

    def test_retries_passthrough(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        config = LLMConfig.from_env()
        assert config.max_retries == 5


# ---------------------------------------------------------------------------
# create_llm_client
# ---------------------------------------------------------------------------

class TestCreateLLMClient:
    def test_no_provider_gives_stub(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        client = create_llm_client()
        assert isinstance(client, StubLLMClient)

    def test_stub_provider_gives_stub(self):
        client = create_llm_client(LLMConfig(provider="stub"))
        assert isinstance(client, StubLLMClient)

    def test_openai_provider_gives_openai_client(self):
        cfg = LLMConfig(provider="openai", api_key="sk-test")
        mock_openai, _, _ = _make_mock_openai()
        with patch("src.llm_client.OpenAILLMClient.__init__", return_value=None):
            client = object.__new__(OpenAILLMClient)
            # Fabrika doğrudan OpenAILLMClient oluşturur; __init__ patched
            with patch("src.llm_client.OpenAILLMClient", return_value=client):
                result = create_llm_client(cfg)
        assert isinstance(result, OpenAILLMClient)

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMConfigurationError, match="Bilinmeyen"):
            create_llm_client(LLMConfig(provider="groq"))

    def test_returns_llm_client_subclass(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        client = create_llm_client()
        assert isinstance(client, LLMClient)

    def test_empty_provider_gives_stub(self):
        client = create_llm_client(LLMConfig(provider=""))
        assert isinstance(client, StubLLMClient)


# ---------------------------------------------------------------------------
# _parse_and_validate_strategy
# ---------------------------------------------------------------------------

class TestParseAndValidateStrategy:
    def test_valid_strategy_returned(self):
        result = _parse_and_validate_strategy(_VALID_STRATEGY_JSON)
        assert result is not None
        assert result["scaling_method"] == "standard"

    def test_all_required_fields_present(self):
        result = _parse_and_validate_strategy(_VALID_STRATEGY_JSON)
        assert result is not None
        required = {
            "imputation_numeric", "imputation_categorical",
            "scaling_method", "encoding_method",
            "outlier_handling", "feature_selection",
        }
        assert required == set(result.keys())

    def test_broken_json_returns_none(self):
        result = _parse_and_validate_strategy("{not valid json}")
        assert result is None

    def test_empty_string_returns_none(self):
        result = _parse_and_validate_strategy("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        result = _parse_and_validate_strategy("   ")
        assert result is None

    def test_missing_field_returns_none(self):
        d = json.loads(_VALID_STRATEGY_JSON)
        del d["scaling_method"]
        result = _parse_and_validate_strategy(json.dumps(d))
        assert result is None

    def test_invalid_enum_value_returns_none(self):
        d = json.loads(_VALID_STRATEGY_JSON)
        d["scaling_method"] = "INVALID_METHOD"
        result = _parse_and_validate_strategy(json.dumps(d))
        assert result is None

    def test_wrong_type_returns_none(self):
        d = json.loads(_VALID_STRATEGY_JSON)
        d["scaling_method"] = 42
        result = _parse_and_validate_strategy(json.dumps(d))
        assert result is None

    def test_none_value_returns_none(self):
        d = json.loads(_VALID_STRATEGY_JSON)
        d["imputation_numeric"] = None
        result = _parse_and_validate_strategy(json.dumps(d))
        assert result is None

    def test_unknown_fields_rejected(self):
        """LLM yanıtında beklenmedik anahtar varsa strateji reddedilmeli."""
        d = json.loads(_VALID_STRATEGY_JSON)
        d["extra_key"] = "surprise"
        result = _parse_and_validate_strategy(json.dumps(d))
        assert result is None

    def test_array_input_returns_none(self):
        result = _parse_and_validate_strategy(json.dumps([1, 2, 3]))
        assert result is None

    def test_null_json_returns_none(self):
        result = _parse_and_validate_strategy("null")
        assert result is None

    def test_valid_all_enum_variants(self):
        for imp_num in ("mean", "median", "knn"):
            d = json.loads(_VALID_STRATEGY_JSON)
            d["imputation_numeric"] = imp_num
            result = _parse_and_validate_strategy(json.dumps(d))
            assert result is not None, f"imputation_numeric={imp_num!r} başarısız"

    def test_valid_scaling_variants(self):
        for scale in ("standard", "robust", "minmax", "none"):
            d = json.loads(_VALID_STRATEGY_JSON)
            d["scaling_method"] = scale
            result = _parse_and_validate_strategy(json.dumps(d))
            assert result is not None, f"scaling_method={scale!r} başarısız"


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_numeric_count(self):
        prompt = _build_prompt(_PROFILE, "classification", set(), 1)
        assert "5" in prompt

    def test_contains_task_type(self):
        prompt = _build_prompt(_PROFILE, "classification", set(), 1)
        assert "classification" in prompt

    def test_contains_regression_type(self):
        prompt = _build_prompt(_PROFILE, "regression", set(), 1)
        assert "regression" in prompt

    def test_tried_signatures_listed(self):
        tried = {"sig-abc", "sig-xyz"}
        prompt = _build_prompt(_PROFILE, "regression", tried, 2)
        assert "sig-abc" in prompt
        assert "sig-xyz" in prompt

    def test_no_tried_signatures_shows_none(self):
        prompt = _build_prompt(_PROFILE, "classification", set(), 1)
        assert "(none)" in prompt

    def test_attempt_number_in_prompt(self):
        prompt = _build_prompt(_PROFILE, "classification", set(), 7)
        assert "7" in prompt

    def test_missing_pct_formatted(self):
        profile = dict(_PROFILE)
        profile["total_missing_pct"] = 0.333
        prompt = _build_prompt(profile, "classification", set(), 1)
        assert "0.33" in prompt

    def test_no_raw_data_values(self):
        # Prompt sadece istatistiksel özet içermeli; raw veri satırı yok
        prompt = _build_prompt(_PROFILE, "classification", set(), 1)
        assert "numeric_column_count" in prompt
        assert "categorical_column_count" in prompt
        assert "total_missing_pct" in prompt


# ---------------------------------------------------------------------------
# OpenAILLMClient — mocked
# ---------------------------------------------------------------------------

class TestOpenAILLMClientBasic:
    def test_valid_response_returned(self):
        client = _make_openai_client()
        result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is not None
        assert result["scaling_method"] == "standard"

    def test_invalid_json_returns_none(self):
        client = _make_openai_client(response_content="{bad json}")
        result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is None

    def test_empty_response_returns_none(self):
        client = _make_openai_client(response_content="")
        result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is None

    def test_fallback_not_activated_by_default(self):
        client = _make_openai_client()
        assert client.fallback_activated is False

    def test_suggest_strategy_calls_chat_completions(self):
        client = _make_openai_client()
        client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert client._client.chat.completions.create.call_count == 1

    def test_api_key_not_in_logs(self, caplog):
        import logging
        client = _make_openai_client()
        with caplog.at_level(logging.DEBUG, logger="src.llm_client"):
            client.suggest_strategy(_PROFILE, "classification", set(), 1)
        for record in caplog.records:
            assert "sk-test" not in record.message


class TestOpenAILLMClientRetry:
    def _make_retrying_client(self, side_effects, max_retries: int = 2):
        cfg = LLMConfig(provider="openai", api_key="sk-test", max_retries=max_retries)
        mock_openai, mock_underlying, mock_response = _make_mock_openai()
        mock_underlying.chat.completions.create.side_effect = side_effects
        client           = object.__new__(OpenAILLMClient)
        client._config   = cfg
        client._openai   = mock_openai
        client._client   = mock_underlying
        client._fallback = None
        client._tls      = threading.local()
        return client, mock_openai, mock_underlying, mock_response

    def test_timeout_triggers_retry(self):
        client, mock_openai, mock_underlying, _ = self._make_retrying_client(
            side_effects=[], max_retries=2
        )
        exc_cls     = mock_openai.APITimeoutError
        mock_msg    = MagicMock(); mock_msg.content = _VALID_STRATEGY_JSON
        mock_ch     = MagicMock(); mock_ch.message = mock_msg
        ok_response = MagicMock(); ok_response.choices = [mock_ch]
        mock_underlying.chat.completions.create.side_effect = [
            exc_cls("timeout"),
            exc_cls("timeout"),
            ok_response,
        ]
        with patch("time.sleep"):
            result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is not None
        assert mock_underlying.chat.completions.create.call_count == 3

    def test_rate_limit_triggers_retry(self):
        client, mock_openai, mock_underlying, mock_response = self._make_retrying_client(
            side_effects=[], max_retries=1
        )
        exc_cls     = mock_openai.RateLimitError
        mock_msg    = MagicMock(); mock_msg.content = _VALID_STRATEGY_JSON
        mock_ch     = MagicMock(); mock_ch.message = mock_msg
        ok_response = MagicMock(); ok_response.choices = [mock_ch]
        mock_underlying.chat.completions.create.side_effect = [
            exc_cls("rate limit"),
            ok_response,
        ]
        with patch("time.sleep"):
            result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is not None
        assert mock_underlying.chat.completions.create.call_count == 2

    def test_server_error_triggers_retry(self):
        client, mock_openai, mock_underlying, mock_response = self._make_retrying_client(
            side_effects=[], max_retries=1
        )
        exc_cls     = mock_openai.InternalServerError
        mock_msg    = MagicMock(); mock_msg.content = _VALID_STRATEGY_JSON
        mock_ch     = MagicMock(); mock_ch.message = mock_msg
        ok_response = MagicMock(); ok_response.choices = [mock_ch]
        mock_underlying.chat.completions.create.side_effect = [
            exc_cls("500"),
            ok_response,
        ]
        with patch("time.sleep"):
            result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is not None

    def test_budget_exhaustion_returns_none(self):
        client, mock_openai, mock_underlying, _ = self._make_retrying_client(
            side_effects=[], max_retries=1
        )
        exc_cls = mock_openai.APITimeoutError
        mock_underlying.chat.completions.create.side_effect = [
            exc_cls("t"), exc_cls("t"),
        ]
        with patch("time.sleep"):
            result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is None

    def test_auth_error_raises_config_error(self):
        client, mock_openai, mock_underlying, _ = self._make_retrying_client(
            side_effects=[], max_retries=0
        )
        exc_cls = mock_openai.AuthenticationError
        mock_underlying.chat.completions.create.side_effect = exc_cls("bad key")
        with pytest.raises(LLMConfigurationError):
            client.suggest_strategy(_PROFILE, "classification", set(), 1)

    def test_auth_error_does_not_retry(self):
        client, mock_openai, mock_underlying, _ = self._make_retrying_client(
            side_effects=[], max_retries=3
        )
        exc_cls = mock_openai.AuthenticationError
        mock_underlying.chat.completions.create.side_effect = exc_cls("bad key")
        with pytest.raises(LLMConfigurationError):
            client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert mock_underlying.chat.completions.create.call_count == 1


class TestOpenAILLMClientFallback:
    def test_fallback_activated_on_budget_exhaustion(self):
        cfg = LLMConfig(
            provider="openai", api_key="sk-test",
            max_retries=0, fallback_enabled=True,
        )
        mock_openai, mock_underlying, _ = _make_mock_openai()
        exc_cls = mock_openai.APITimeoutError
        mock_underlying.chat.completions.create.side_effect = exc_cls("timeout")

        client           = object.__new__(OpenAILLMClient)
        client._config   = cfg
        client._openai   = mock_openai
        client._client   = mock_underlying
        client._fallback = StubLLMClient()
        client._tls      = threading.local()

        with patch("time.sleep"):
            result = client.suggest_strategy(
                {"numeric_column_count": 3, "categorical_column_count": 1, "total_missing_pct": 0.0},
                "classification", set(), 1,
            )
        assert client.fallback_activated is True
        assert result is not None  # StubLLMClient bir strateji döndürür

    def test_fallback_disabled_returns_none(self):
        cfg = LLMConfig(
            provider="openai", api_key="sk-test",
            max_retries=0, fallback_enabled=False,
        )
        mock_openai, mock_underlying, _ = _make_mock_openai()
        exc_cls = mock_openai.APITimeoutError
        mock_underlying.chat.completions.create.side_effect = exc_cls("timeout")

        client           = object.__new__(OpenAILLMClient)
        client._config   = cfg
        client._openai   = mock_openai
        client._client   = mock_underlying
        client._fallback = None
        client._tls      = threading.local()

        with patch("time.sleep"):
            result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is None
        assert client.fallback_activated is False

    def test_subsequent_calls_use_fallback_after_activation(self):
        cfg = LLMConfig(
            provider="openai", api_key="sk-test",
            max_retries=0, fallback_enabled=True,
        )
        mock_openai, mock_underlying, _ = _make_mock_openai()
        exc_cls = mock_openai.APITimeoutError
        mock_underlying.chat.completions.create.side_effect = exc_cls("timeout")

        client           = object.__new__(OpenAILLMClient)
        client._config   = cfg
        client._openai   = mock_openai
        client._client   = mock_underlying
        client._fallback = StubLLMClient()
        client._tls      = threading.local()

        profile = {"numeric_column_count": 2, "categorical_column_count": 0, "total_missing_pct": 0.0}

        with patch("time.sleep"):
            client.suggest_strategy(profile, "classification", set(), 1)

        assert client.fallback_activated is True
        # İkinci çağrı doğrudan fallback'e gitmeli; _client.create çağrılmaz
        call_count_before = mock_underlying.chat.completions.create.call_count
        client.suggest_strategy(profile, "classification", set(), 2)
        assert mock_underlying.chat.completions.create.call_count == call_count_before


# ---------------------------------------------------------------------------
# AI-03: oturum yönetimi ve bütçe zorunluluğu
# ---------------------------------------------------------------------------

class TestOpenAILLMClientSessionManagement:
    def test_begin_session_resets_counter(self):
        client = _make_openai_client()
        client.begin_session(budget=15)
        client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert client.session_provider_calls == 1
        client.begin_session(budget=15)
        assert client.session_provider_calls == 0

    def test_begin_session_resets_budget_exhausted(self):
        client = _make_openai_client(
            cfg=LLMConfig(provider="openai", api_key="sk-test", max_retries=0)
        )
        client.begin_session(budget=1)
        client.suggest_strategy(_PROFILE, "classification", set(), 1)
        # İkinci çağrı bütçeyi tüketir
        client.suggest_strategy(_PROFILE, "classification", set(), 2)
        assert client.budget_exhausted is True
        client.begin_session(budget=15)
        assert client.budget_exhausted is False

    def test_provider_calls_counted_per_sdk_call(self):
        client = _make_openai_client()
        client.begin_session(budget=5)
        client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert client.session_provider_calls == 1

    def test_sixteenth_call_blocked(self):
        """16. çağrı SDK'ya ulaşmamalı."""
        client = _make_openai_client(
            cfg=LLMConfig(provider="openai", api_key="sk-test", max_retries=0)
        )
        client.begin_session(budget=15)
        for i in range(15):
            client.suggest_strategy(_PROFILE, "classification", set(), i + 1)
        result = client.suggest_strategy(_PROFILE, "classification", set(), 16)
        assert result is None
        assert client.budget_exhausted is True
        assert client._client.chat.completions.create.call_count == 15

    def test_no_backoff_when_budget_exhausted(self):
        """Bütçe tükendiğinde retry bekleme uygulanmamalı."""
        client = _make_openai_client(
            cfg=LLMConfig(provider="openai", api_key="sk-test", max_retries=5)
        )
        client.begin_session(budget=1)
        exc_cls = client._openai.APITimeoutError
        client._client.chat.completions.create.side_effect = exc_cls("timeout")
        with patch("time.sleep") as mock_sleep:
            client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert mock_sleep.call_count == 0

    def test_budget_exhausted_does_not_trigger_fallback(self):
        """Bütçe tükenince stub fallback devreye GİRMEMELİ."""
        cfg = LLMConfig(
            provider="openai", api_key="sk-test",
            max_retries=0, fallback_enabled=True,
        )
        client           = object.__new__(OpenAILLMClient)
        client._config   = cfg
        client._openai   = _make_mock_openai()[0]
        client._client   = _make_mock_openai()[1]
        client._fallback = StubLLMClient()
        client._tls      = threading.local()
        client.begin_session(budget=0)  # bütçe zaten tükenmiş
        result = client.suggest_strategy(_PROFILE, "classification", set(), 1)
        assert result is None
        assert client.fallback_activated is False

    def test_concurrent_sessions_isolated(self):
        """Aynı istemciyi kullanan iki thread birbirinin bütçesini bozmamalı."""
        client  = _make_openai_client(
            cfg=LLMConfig(provider="openai", api_key="sk-test", max_retries=0)
        )
        t1_results: list = []
        t2_results: list = []
        barrier = threading.Barrier(2)

        def run_t1():
            client.begin_session(budget=3)
            barrier.wait()
            for i in range(4):  # 4 girişim; bütçe yalnızca 3'e izin verir
                client.suggest_strategy(_PROFILE, "classification", set(), i + 1)
            t1_results.append((client.session_provider_calls, client.budget_exhausted))

        def run_t2():
            client.begin_session(budget=10)
            barrier.wait()
            client.suggest_strategy(_PROFILE, "classification", set(), 1)
            t2_results.append((client.session_provider_calls, client.budget_exhausted))

        th1 = threading.Thread(target=run_t1)
        th2 = threading.Thread(target=run_t2)
        th1.start(); th2.start()
        th1.join();  th2.join()

        t1_calls, t1_exhausted = t1_results[0]
        t2_calls, t2_exhausted = t2_results[0]

        assert t1_calls == 3           # T1'in bütçesi 3
        assert t1_exhausted is True
        assert t2_calls == 1           # T2, T1'den etkilenmemiş
        assert t2_exhausted is False


# ---------------------------------------------------------------------------
# AI-03: SDK otomatik retry devre dışı
# ---------------------------------------------------------------------------

class TestSDKAutoRetryDisabled:
    def test_max_retries_zero_passed_to_openai_constructor(self):
        """OpenAI() çağrısında max_retries=0 iletilmeli (SDK retry devre dışı)."""
        mock_openai      = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()
        cfg = LLMConfig(provider="openai", api_key="sk-test")
        with patch.dict("sys.modules", {"openai": mock_openai}):
            OpenAILLMClient(cfg)
        assert mock_openai.OpenAI.call_args.kwargs.get("max_retries") == 0
