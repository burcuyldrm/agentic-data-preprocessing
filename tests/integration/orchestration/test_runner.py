"""
src.orchestration.runner — Hedefli entegrasyon testleri.

Mevcut test_orchestrator.py testleri korunur; bu dosya runner'a özgü
davranışları (geçersiz strateji, tekrar, zaman sınırı, mod farkları,
ardışık başarısızlık) doğrular ve yeni import yollarını kapsar.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.llm_client import LLMClient
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
# Stub LLM istemcileri
# ---------------------------------------------------------------------------

class _NullLLMClient(LLMClient):
    """Hiçbir strateji öneremeyen stub — döngüyü ilk denemede sonlandırır."""
    def suggest_strategy(self, profile_summary, task_type, tried_signatures, attempt_num):
        return None


class _AlwaysSameLLMClient(LLMClient):
    """Her denemede aynı geçerli stratejiyi döndürür — tekrar kontrolünü tetikler."""
    _STRATEGY = {
        "imputation_numeric":     "median",
        "imputation_categorical": "most_frequent",
        "scaling_method":         "standard",
        "encoding_method":        "onehot",
        "outlier_handling":       "none",
        "feature_selection":      "none",
    }
    def suggest_strategy(self, profile_summary, task_type, tried_signatures, attempt_num):
        return dict(self._STRATEGY)


class _AlwaysInvalidLLMClient(LLMClient):
    """
    Her denemede farklı (geçersiz) strateji döndürür.
    consecutive_fails sayacını doldurmak için farklı imzalar üretir.
    """
    def __init__(self):
        self._n = 0

    def suggest_strategy(self, profile_summary, task_type, tried_signatures, attempt_num):
        self._n += 1
        return {
            "imputation_numeric":     f"INVALID_{self._n}",  # geçersiz değer
            "imputation_categorical": "most_frequent",
            "scaling_method":         "standard",
            "encoding_method":        "onehot",
            "outlier_handling":       "none",
            "feature_selection":      "none",
        }


# ---------------------------------------------------------------------------
# 1. Import yolları ve kimlik
# ---------------------------------------------------------------------------

class TestRunnerImport:
    def test_importable_from_runner_module(self):
        from src.orchestration.runner import Orchestrator as RunnerOrch
        assert RunnerOrch is not None

    def test_runner_same_as_package_init(self):
        from src.orchestration import Orchestrator as PkgOrch
        from src.orchestration.runner import Orchestrator as RunnerOrch
        assert RunnerOrch is PkgOrch

    def test_runner_same_as_shim(self):
        from src.orchestration.runner import Orchestrator as RunnerOrch
        from src.orchestrator import Orchestrator as ShimOrch
        assert RunnerOrch is ShimOrch


# ---------------------------------------------------------------------------
# 2. Temel davranışlar
# ---------------------------------------------------------------------------

class TestRunnerBasicBehavior:
    def _cfg(self, max_attempts: int = 1):
        return OrchestratorConfig(max_attempts=max_attempts, n_cv_splits=2)

    def test_classification_completed(self, tmp_path):
        csv    = _make_csv(tmp_path, task="classification")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_regression_completed(self, tmp_path):
        csv    = _make_csv(tmp_path, task="regression")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_missing_csv_returns_failed(self, tmp_path):
        result = Orchestrator().run("/nonexistent/path.csv", "target", str(tmp_path / "db"))
        assert result.status == "failed"

    def test_missing_csv_no_exception(self, tmp_path):
        result = Orchestrator().run("/nonexistent/path.csv", "target", str(tmp_path / "db"))
        assert isinstance(result, __import__("src.orchestration.models", fromlist=["AnalysisResult"]).AnalysisResult)

    def test_missing_csv_error_message_set(self, tmp_path):
        result = Orchestrator().run("/nonexistent/path.csv", "target", str(tmp_path / "db"))
        assert result.error_message is not None


# ---------------------------------------------------------------------------
# 3. Strateji seçimi senaryoları
# ---------------------------------------------------------------------------

class TestRunnerStrategyHandling:
    def _cfg(self, **kwargs):
        return OrchestratorConfig(n_cv_splits=2, **kwargs)

    def test_null_llm_returns_failed(self, tmp_path):
        """LLM None döndürürse ve bellekte strateji yoksa sonuç 'failed' olmalı.

        Mod C: LLM döngüsü kullanır, NullLLM ilk denemede None döndürür → failed.
        (Mod A bu testi test etmeye uygun değil — Mod A LLM kullanmaz.)
        """
        csv = _make_csv(tmp_path)
        cfg = self._cfg(max_attempts=3, experiment_mode="C")
        result = Orchestrator(llm_client=_NullLLMClient(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "failed"
        assert len(result.attempts) == 0

    def test_null_llm_no_exception(self, tmp_path):
        """NullLLM exception fırlatmaz, 'failed' durumuyla döner.

        Mod C kullanılır — Mod A LLM'i hiç çağırmaz, bu robustness testini
        doğru şekilde temsil etmez.
        """
        csv = _make_csv(tmp_path)
        cfg = self._cfg(max_attempts=1, experiment_mode="C")
        result = Orchestrator(llm_client=_NullLLMClient(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert isinstance(result.status, str)

    def test_duplicate_strategy_one_attempt_only(self, tmp_path):
        """Aynı strateji tekrar önerilse yalnızca bir kez değerlendirilmeli.

        Mod C kullanılır — Mod A LLM döngüsü çalıştırmaz, tekrar kontrolünü
        doğrudan test etmez.
        """
        csv = _make_csv(tmp_path)
        cfg = self._cfg(max_attempts=5, experiment_mode="C")
        result = Orchestrator(llm_client=_AlwaysSameLLMClient(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "completed"
        assert len(result.attempts) == 1

    def test_consecutive_failures_stops_loop(self, tmp_path):
        """max_consecutive_failures dolunca döngü erken durmalı.

        Mod C kullanılır — Mod A LLM döngüsü çalıştırmaz, bu senaryoyu
        test etmez.
        """
        csv = _make_csv(tmp_path)
        cfg = self._cfg(
            max_attempts=10,
            max_consecutive_failures=2,
            experiment_mode="C",
        )
        result = Orchestrator(llm_client=_AlwaysInvalidLLMClient(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "failed"
        # 2 geçersiz deneme kaydedilmeli
        assert len(result.attempts) == 2
        assert all(r.status == "invalid" for r in result.attempts)


# ---------------------------------------------------------------------------
# 4. Zaman sınırı ve deneme sayısı
# ---------------------------------------------------------------------------

class TestRunnerTimeLimitAndAttempts:
    def test_time_limit_zero_returns_failed(self, tmp_path):
        """time_limit_seconds=0.0 ise döngü ilk iterasyonda kırılmalı."""
        csv = _make_csv(tmp_path)
        cfg = OrchestratorConfig(
            max_attempts=5, n_cv_splits=2, time_limit_seconds=0.0
        )
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "failed"
        assert len(result.attempts) == 0

    def test_max_attempts_one_records_one_attempt(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=1, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert len(result.attempts) == 1

    def test_max_attempts_not_exceeded(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=3, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert len(result.attempts) <= 3


# ---------------------------------------------------------------------------
# 5. Deney modları
# ---------------------------------------------------------------------------

class TestRunnerExperimentModes:
    def _cfg(self, mode: str):
        return OrchestratorConfig(max_attempts=1, n_cv_splits=2, experiment_mode=mode)

    def test_mode_a_completes(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg("A")).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_mode_b_completes(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg("B")).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_mode_c_completes(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg("C")).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_mode_d_completes(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg("D")).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_mode_a_does_not_use_memory_strategies(self, tmp_path):
        """
        Mod A LLM veya bellek kullanmaz; NullLLM verilse bile baseline
        stratejiyle tamamlanır.

        Yeni deney sözleşmesi (AI-01): Mod A sabit baseline kullanır.
        Önceki test, Mod A'nın LLM döngüsü kullandığını varsayıyordu;
        bu davranış değişti.
        """
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=3, n_cv_splits=2, experiment_mode="A")
        result = Orchestrator(llm_client=_NullLLMClient(), config=cfg).run(
            csv, "target", str(tmp_path / "db")
        )
        assert result.status == "completed"
        assert len(result.attempts) == 1
        assert result.total_llm_calls == 0


# ---------------------------------------------------------------------------
# 6. Veritabanı kayıtları
# ---------------------------------------------------------------------------

class TestRunnerDBRecords:
    def _cfg(self):
        return OrchestratorConfig(max_attempts=1, n_cv_splits=2)

    def test_analysis_run_recorded_in_db(self, tmp_path):
        csv    = _make_csv(tmp_path)
        db     = str(tmp_path / "run.sqlite")
        result = Orchestrator(config=self._cfg()).run(csv, "target", db)
        conn   = sqlite3.connect(db)
        row    = conn.execute(
            "SELECT status FROM analysis_runs WHERE id = ?", (result.run_id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "completed"

    def test_evaluation_attempts_recorded_in_db(self, tmp_path):
        csv    = _make_csv(tmp_path)
        db     = str(tmp_path / "run.sqlite")
        result = Orchestrator(config=self._cfg()).run(csv, "target", db)
        conn   = sqlite3.connect(db)
        rows   = conn.execute(
            "SELECT id FROM evaluation_attempts WHERE analysis_run_id = ?",
            (result.run_id,),
        ).fetchall()
        conn.close()
        assert len(rows) >= 1

    def test_failed_run_recorded_in_db(self, tmp_path):
        """Başarısız çalışma 'failed' durumuyla veritabanına yazılmalı.

        Mod C kullanılır — Mod A LLM döngüsü çalıştırmadığı için
        AlwaysInvalidLLMClient bu senaryoyu Mod A'da test edemez.
        """
        csv = _make_csv(tmp_path)
        db  = str(tmp_path / "run.sqlite")
        cfg = OrchestratorConfig(
            max_attempts=3, n_cv_splits=2,
            max_consecutive_failures=2, experiment_mode="C"
        )
        result = Orchestrator(llm_client=_AlwaysInvalidLLMClient(), config=cfg).run(
            csv, "target", db
        )
        assert result.run_id is not None
        conn = sqlite3.connect(db)
        row  = conn.execute(
            "SELECT status FROM analysis_runs WHERE id = ?", (result.run_id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "failed"
