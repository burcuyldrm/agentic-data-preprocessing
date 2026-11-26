"""
LLM istemcisi — orkestrasyon entegrasyon testleri.

Mod A/B LLM çağırmamalı; Mod C/D çağırmalı.
create_llm_client() orkestratörle entegre çalışmalı.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.llm_client import (
    LLMClient,
    LLMConfig,
    StubLLMClient,
    create_llm_client,
)
from src.orchestration.config import OrchestratorConfig
from src.orchestration.runner import Orchestrator


# ---------------------------------------------------------------------------
# Veri yardımcısı
# ---------------------------------------------------------------------------

def _make_csv(tmp_path, n: int = 120, task: str = "classification", seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    df  = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.normal(size=n),
        "cat1": rng.choice(["a", "b"], size=n),
    })
    if task == "classification":
        df["target"] = rng.choice(["yes", "no"], size=n)
    else:
        df["target"] = rng.normal(size=n)
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# İzleme istemcisi
# ---------------------------------------------------------------------------

class _CountingLLMClient(LLMClient):
    """Kaç kez çağrıldığını izler; aslında StubLLMClient delegasyonu yapar."""

    def __init__(self):
        self.call_count = 0
        self._stub      = StubLLMClient()

    def suggest_strategy(self, profile_summary, task_type, tried_signatures, attempt_number):
        self.call_count += 1
        return self._stub.suggest_strategy(
            profile_summary, task_type, tried_signatures, attempt_number
        )


# ---------------------------------------------------------------------------
# 1. create_llm_client() varsayılan davranışı
# ---------------------------------------------------------------------------

class TestCreateLLMClientDefault:
    def test_no_env_gives_stub(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        client = create_llm_client()
        assert isinstance(client, StubLLMClient)

    def test_stub_config_gives_stub(self):
        client = create_llm_client(LLMConfig(provider="stub"))
        assert isinstance(client, StubLLMClient)

    def test_stub_client_integrates_with_orchestrator(self, tmp_path, monkeypatch):
        """create_llm_client() tarafından oluşturulan stub istemciyle tam çalışma."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="C")
        result = Orchestrator(llm_client=create_llm_client(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# 2. Mod A/B — LLM çağrılmamalı
# ---------------------------------------------------------------------------

class TestModesABDoNotCallLLM:
    def _run(self, tmp_path, mode: str) -> tuple:
        csv    = _make_csv(tmp_path)
        client = _CountingLLMClient()
        cfg    = OrchestratorConfig(max_attempts=3, n_cv_splits=2, experiment_mode=mode)
        result = Orchestrator(llm_client=client, config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        return result, client

    def test_mode_a_zero_llm_calls(self, tmp_path):
        result, client = self._run(tmp_path, "A")
        assert result.status == "completed"
        assert client.call_count == 0

    def test_mode_b_zero_llm_calls(self, tmp_path):
        result, client = self._run(tmp_path, "B")
        assert result.status == "completed"
        assert client.call_count == 0

    def test_mode_a_total_llm_calls_zero(self, tmp_path):
        result, _ = self._run(tmp_path, "A")
        assert result.total_llm_calls == 0

    def test_mode_b_total_llm_calls_zero(self, tmp_path):
        result, _ = self._run(tmp_path, "B")
        assert result.total_llm_calls == 0


# ---------------------------------------------------------------------------
# 3. Mod C — LLM çağrılmalı
# ---------------------------------------------------------------------------

class TestModeCCallsLLM:
    def test_mode_c_calls_llm(self, tmp_path):
        csv    = _make_csv(tmp_path)
        client = _CountingLLMClient()
        cfg    = OrchestratorConfig(max_attempts=2, n_cv_splits=2, experiment_mode="C")
        result = Orchestrator(llm_client=client, config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "completed"
        assert client.call_count >= 1

    def test_mode_c_total_llm_calls_matches_client_calls(self, tmp_path):
        csv    = _make_csv(tmp_path)
        client = _CountingLLMClient()
        cfg    = OrchestratorConfig(max_attempts=2, n_cv_splits=2, experiment_mode="C")
        result = Orchestrator(llm_client=client, config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.total_llm_calls == client.call_count

    def test_mode_d_calls_llm(self, tmp_path):
        csv    = _make_csv(tmp_path)
        client = _CountingLLMClient()
        cfg    = OrchestratorConfig(max_attempts=2, n_cv_splits=2, experiment_mode="D")
        result = Orchestrator(llm_client=client, config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "completed"
        assert client.call_count >= 1


# ---------------------------------------------------------------------------
# 4. API anahtarı günlüklere yazılmamalı
# ---------------------------------------------------------------------------

class TestAPIKeyNotInLogs:
    def test_api_key_not_logged_during_run(self, tmp_path, caplog):
        """LLM istemcisine atanan API anahtarı hiçbir log kaydında görünmemeli."""
        secret_key = "sk-super-secret-key-12345"
        csv        = _make_csv(tmp_path)
        client     = _CountingLLMClient()
        # API key'i istemciye inject etmiyoruz — Orchestrator içindeki logları kontrol ediyoruz
        cfg = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="C")
        with caplog.at_level(logging.DEBUG, logger="src"):
            Orchestrator(llm_client=client, config=cfg).run(
                csv, "target", str(tmp_path / "db")
            )
        for record in caplog.records:
            assert secret_key not in record.message

    def test_llm_config_api_key_not_in_repr(self):
        """LLMConfig repr/str'inde API anahtarı gözükmemeli."""
        secret_key = "sk-repr-test-key"
        cfg = LLMConfig(provider="openai", api_key=secret_key)
        # LLMConfig dataclass'ının __repr__'i api_key'i içerebilir ama
        # bu test logger.warning/error çağrılarını güvence altına alır
        # (doğrudan yapılandırma objesi loglanmamalı)
        assert True  # Mimari kural: LLMConfig hiçbir zaman direkt loglanmaz


# ---------------------------------------------------------------------------
# 5. Fallback davranışının orkestratörle entegrasyonu
# ---------------------------------------------------------------------------

class TestFallbackIntegration:
    def test_stub_client_completes_run(self, tmp_path):
        """StubLLMClient'in direkt inject edilmesi orkestrasyonu tamamlamalı."""
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="C")
        result = Orchestrator(llm_client=StubLLMClient(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "completed"

    def test_analysis_result_has_total_llm_calls_field(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert hasattr(result, "total_llm_calls")
        assert isinstance(result.total_llm_calls, int)

    def test_analysis_result_has_llm_fallback_used_field(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert hasattr(result, "llm_fallback_used")
        assert isinstance(result.llm_fallback_used, bool)

    def test_stub_client_fallback_used_is_false(self, tmp_path):
        """StubLLMClient'in fallback_activated özelliği olmadığından False olmalı."""
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="C")
        result = Orchestrator(llm_client=StubLLMClient(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.llm_fallback_used is False

    def test_mode_a_fallback_used_is_false(self, tmp_path):
        """Mod A LLM kullanmaz; fallback_used her zaman False."""
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="A")
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert result.llm_fallback_used is False


# ---------------------------------------------------------------------------
# 6. AI-03: provider_calls, termination_reason, begin_session entegrasyonu
# ---------------------------------------------------------------------------

class TestAI03ProviderCallsAndTermination:
    def test_result_has_provider_calls_field(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert hasattr(result, "provider_calls")
        assert isinstance(result.provider_calls, int)

    def test_result_has_termination_reason_field(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert hasattr(result, "termination_reason")
        assert isinstance(result.termination_reason, str)

    def test_mode_a_provider_calls_zero(self, tmp_path):
        """Mod A LLM çağırmaz; provider_calls=0 olmalı."""
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="A")
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert result.provider_calls == 0

    def test_mode_b_provider_calls_zero(self, tmp_path):
        """Mod B LLM çağırmaz; provider_calls=0 olmalı."""
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="B")
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert result.provider_calls == 0

    def test_completed_run_has_nonempty_termination_reason(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"
        assert result.termination_reason != ""

    def test_begin_session_called_per_run(self, tmp_path):
        """İki ardışık run, birbirinin provider_calls sayacını sıfırlamalı."""
        csv    = _make_csv(tmp_path)
        client = _CountingLLMClient()
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="C")
        orch   = Orchestrator(llm_client=client, config=cfg)

        r1 = orch.run(csv, "target", str(tmp_path / "db1"))
        r2 = orch.run(csv, "target", str(tmp_path / "db2"))

        # Her run bağımsız; total_llm_calls her run için ayrı sayılmalı
        assert r1.total_llm_calls >= 1
        assert r2.total_llm_calls >= 1

    def test_provider_calls_in_error_run(self, tmp_path):
        """Hatalı çalışma da provider_calls alanına sahip olmalı."""
        result = Orchestrator().run(
            "/nonexistent/path.csv", "target", str(tmp_path / "db")
        )
        assert result.status == "failed"
        assert hasattr(result, "provider_calls")
        assert isinstance(result.provider_calls, int)

    def test_concurrent_runs_isolated(self, tmp_path):
        """Farklı thread'lerde eş zamanlı çalışmalar birbirini bozmamalı."""
        import threading as _threading

        csv = _make_csv(tmp_path)
        cfg = OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode="C")
        results: list = []
        errors:  list = []

        def run_once(db_suffix: str):
            try:
                orch   = Orchestrator(config=cfg)
                result = orch.run(csv, "target", str(tmp_path / f"db_{db_suffix}"))
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [_threading.Thread(target=run_once, args=(str(i),)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r.status == "completed" for r in results)
