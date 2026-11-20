"""
src.orchestration alt modüllerinin doğrudan import ve davranış testleri.

Bu testler:
  - Yeni modül yollarının çalıştığını doğrular.
  - src.orchestrator shim üzerinden geriye dönük uyumluluğu doğrular.
  - Mevcut test_orchestrator.py testleriyle çakışmaz; davranışı
    yeni yollar üzerinden ayrıca örnekler.
"""

import hashlib
import os
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Doğrudan modül import testleri
# ---------------------------------------------------------------------------

class TestDirectImports:
    def test_utils_importable(self):
        from src.orchestration.utils import _file_hash, _is_better
        assert callable(_file_hash)
        assert callable(_is_better)

    def test_config_importable(self):
        from src.orchestration.config import OrchestratorConfig
        assert OrchestratorConfig is not None

    def test_models_importable(self):
        from src.orchestration.models import AnalysisResult, AttemptRecord
        assert AttemptRecord is not None
        assert AnalysisResult is not None

    def test_package_init_importable(self):
        from src.orchestration import (
            AnalysisResult,
            AttemptRecord,
            OrchestratorConfig,
            _file_hash,
            _is_better,
        )
        assert all(x is not None for x in [
            AnalysisResult, AttemptRecord, OrchestratorConfig, _file_hash, _is_better,
        ])

    def test_shim_exports_same_objects(self):
        from src.orchestration.config import OrchestratorConfig as New
        from src.orchestration.models import AnalysisResult as NewAR
        from src.orchestration.models import AttemptRecord as NewAttempt
        from src.orchestration.utils import _is_better as new_is_better
        from src.orchestrator import AnalysisResult as Old
        from src.orchestrator import AttemptRecord as OldAttempt
        from src.orchestrator import OrchestratorConfig as OldCfg
        from src.orchestrator import _is_better as old_is_better

        assert New is OldCfg
        assert NewAR is Old
        assert NewAttempt is OldAttempt
        assert new_is_better is old_is_better


# ---------------------------------------------------------------------------
# src.orchestration.utils — _is_better
# ---------------------------------------------------------------------------

class TestIsBetterViaNewPath:
    def test_classification_higher_wins(self):
        from src.orchestration.utils import _is_better
        assert _is_better(0.9, 0.8, "classification") is True

    def test_classification_lower_loses(self):
        from src.orchestration.utils import _is_better
        assert _is_better(0.7, 0.8, "classification") is False

    def test_classification_equal_not_better(self):
        from src.orchestration.utils import _is_better
        assert _is_better(0.8, 0.8, "classification") is False

    def test_regression_lower_wins(self):
        from src.orchestration.utils import _is_better
        assert _is_better(0.5, 0.8, "regression") is True

    def test_regression_higher_loses(self):
        from src.orchestration.utils import _is_better
        assert _is_better(1.0, 0.8, "regression") is False

    def test_regression_equal_not_better(self):
        from src.orchestration.utils import _is_better
        assert _is_better(0.8, 0.8, "regression") is False


# ---------------------------------------------------------------------------
# src.orchestration.utils — _file_hash
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_returns_64_char_hex(self, tmp_path):
        from src.orchestration.utils import _file_hash
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        result = _file_hash(str(f))
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self, tmp_path):
        from src.orchestration.utils import _file_hash
        f = tmp_path / "data.csv"
        f.write_text("x,y\n3,4\n")
        assert _file_hash(str(f)) == _file_hash(str(f))

    def test_matches_stdlib_sha256(self, tmp_path):
        from src.orchestration.utils import _file_hash
        content = b"col1,col2\n1,2\n3,4\n"
        f = tmp_path / "check.csv"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _file_hash(str(f)) == expected

    def test_different_content_different_hash(self, tmp_path):
        from src.orchestration.utils import _file_hash
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("a,b\n1,2\n")
        f2.write_text("a,b\n1,3\n")
        assert _file_hash(str(f1)) != _file_hash(str(f2))


# ---------------------------------------------------------------------------
# src.orchestration.config — OrchestratorConfig
# ---------------------------------------------------------------------------

class TestOrchestratorConfigViaNewPath:
    def test_defaults(self):
        from src.orchestration.config import OrchestratorConfig
        cfg = OrchestratorConfig()
        assert cfg.max_attempts == 5
        assert cfg.max_consecutive_failures == 2
        assert cfg.time_limit_seconds == 600.0
        assert cfg.random_seed == 42
        assert cfg.experiment_mode == "C"
        assert cfg.n_cv_splits == 5

    def test_custom_values(self):
        from src.orchestration.config import OrchestratorConfig
        cfg = OrchestratorConfig(max_attempts=3, random_seed=0, experiment_mode="D")
        assert cfg.max_attempts == 3
        assert cfg.random_seed == 0
        assert cfg.experiment_mode == "D"


# ---------------------------------------------------------------------------
# src.orchestration.models — AttemptRecord, AnalysisResult
# ---------------------------------------------------------------------------

class TestModelsViaNewPath:
    def test_attempt_record_required_fields(self):
        from src.orchestration.models import AttemptRecord
        rec = AttemptRecord(
            attempt_number=1,
            strategy={"imputation_numeric": "median"},
            strategy_id=42,
            cv_primary_score=0.85,
            cv_secondary_score=0.80,
            primary_metric="macro_f1",
            duration_seconds=1.5,
            status="success",
        )
        assert rec.attempt_number == 1
        assert rec.status == "success"
        assert rec.error_message is None

    def test_attempt_record_optional_error(self):
        from src.orchestration.models import AttemptRecord
        rec = AttemptRecord(
            attempt_number=2,
            strategy={},
            strategy_id=0,
            cv_primary_score=None,
            cv_secondary_score=None,
            primary_metric="rmse",
            duration_seconds=0.0,
            status="failed",
            error_message="hata",
        )
        assert rec.error_message == "hata"

    def test_analysis_result_defaults(self):
        from src.orchestration.models import AnalysisResult
        res = AnalysisResult(
            status="completed",
            task_type="classification",
            best_strategy={"imputation_numeric": "median"},
            best_cv_score=0.90,
            primary_metric="macro_f1",
            final_test_scores={"final_test_primary_score": 0.88},
        )
        assert res.attempts == []
        assert res.total_duration_secs == 0.0
        assert res.run_id is None
        assert res.error_message is None

    def test_analysis_result_failed_state(self):
        from src.orchestration.models import AnalysisResult
        res = AnalysisResult(
            status="failed",
            task_type="",
            best_strategy=None,
            best_cv_score=None,
            primary_metric=None,
            final_test_scores=None,
            error_message="beklenmedik hata",
        )
        assert res.status == "failed"
        assert res.error_message == "beklenmedik hata"
