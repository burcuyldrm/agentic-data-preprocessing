"""
FR-07 — Orkestrasyon ve yardımcı fonksiyon testleri.

Kabul kriterleri (FR-07):
  - Sınıflandırma ve regresyon veri setleriyle orkestrasyon tamamlanır.
  - Bellek ve rapor kayıtları veritabanında doğrulanabilir.
  - Geçersiz CSV yolu 'failed' durumu döndürür; exception fırlatmaz.
  - StubLLMClient: 6 benzersiz aday önerir, hepsi denenince None döner.
  - save_analysis_report: analysis_reports tablosuna doğru yazar;
    aynı run_id için ikinci çağrı IntegrityError fırlatır.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.orchestrator import AnalysisResult, Orchestrator, OrchestratorConfig, _is_better


# ---------------------------------------------------------------------------
# Veri yardımcıları
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
# TestIsBetter
# ---------------------------------------------------------------------------

class TestIsBetter:
    def test_classification_higher_is_better(self):
        assert _is_better(0.9, 0.8, "classification") is True

    def test_classification_lower_is_not_better(self):
        assert _is_better(0.7, 0.8, "classification") is False

    def test_regression_lower_is_better(self):
        assert _is_better(0.5, 0.8, "regression") is True

    def test_regression_higher_is_not_better(self):
        assert _is_better(1.0, 0.8, "regression") is False

    def test_equal_scores_not_better_classification(self):
        assert _is_better(0.8, 0.8, "classification") is False

    def test_equal_scores_not_better_regression(self):
        assert _is_better(0.8, 0.8, "regression") is False


# ---------------------------------------------------------------------------
# TestOrchestratorConfig
# ---------------------------------------------------------------------------

class TestOrchestratorConfig:
    def test_default_max_attempts(self):
        assert OrchestratorConfig().max_attempts == 5

    def test_default_max_consecutive_failures(self):
        assert OrchestratorConfig().max_consecutive_failures == 2

    def test_default_time_limit(self):
        assert OrchestratorConfig().time_limit_seconds == 600.0

    def test_default_random_seed(self):
        assert OrchestratorConfig().random_seed == 42

    def test_default_experiment_mode(self):
        assert OrchestratorConfig().experiment_mode == "C"

    def test_default_n_cv_splits(self):
        assert OrchestratorConfig().n_cv_splits == 5

    def test_custom_max_attempts(self):
        cfg = OrchestratorConfig(max_attempts=3)
        assert cfg.max_attempts == 3

    def test_custom_random_seed(self):
        cfg = OrchestratorConfig(random_seed=0)
        assert cfg.random_seed == 0


# ---------------------------------------------------------------------------
# TestOrchestratorRun
# ---------------------------------------------------------------------------

class TestOrchestratorRun:
    def _cfg(self, max_attempts: int = 1):
        return OrchestratorConfig(max_attempts=max_attempts, n_cv_splits=2)

    def test_classification_status_completed(self, tmp_path):
        csv    = _make_csv(tmp_path, task="classification")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_classification_task_type(self, tmp_path):
        csv    = _make_csv(tmp_path, task="classification")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.task_type == "classification"

    def test_regression_status_completed(self, tmp_path):
        csv    = _make_csv(tmp_path, task="regression")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.status == "completed"

    def test_regression_task_type(self, tmp_path):
        csv    = _make_csv(tmp_path, task="regression")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.task_type == "regression"

    def test_best_strategy_not_none(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.best_strategy is not None

    def test_best_cv_score_not_none(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.best_cv_score is not None

    def test_final_test_scores_present(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.final_test_scores is not None
        assert "final_test_primary_score" in result.final_test_scores

    def test_at_least_one_attempt_recorded(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert len(result.attempts) >= 1

    def test_max_attempts_not_exceeded(self, tmp_path):
        csv    = _make_csv(tmp_path)
        cfg    = OrchestratorConfig(max_attempts=3, n_cv_splits=2)
        result = Orchestrator(config=cfg).run(csv, "target", str(tmp_path / "db"))
        assert len(result.attempts) <= 3

    def test_run_id_not_none(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.run_id is not None

    def test_analysis_run_status_completed_in_db(self, tmp_path):
        csv    = _make_csv(tmp_path)
        db     = str(tmp_path / "db.sqlite")
        result = Orchestrator(config=self._cfg()).run(csv, "target", db)
        conn   = sqlite3.connect(db)
        row    = conn.execute(
            "SELECT status FROM analysis_runs WHERE id = ?", (result.run_id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "completed"

    def test_analysis_report_in_db(self, tmp_path):
        csv    = _make_csv(tmp_path)
        db     = str(tmp_path / "db.sqlite")
        result = Orchestrator(config=self._cfg()).run(csv, "target", db)
        conn   = sqlite3.connect(db)
        row    = conn.execute(
            "SELECT id FROM analysis_reports WHERE analysis_run_id = ?",
            (result.run_id,),
        ).fetchone()
        conn.close()
        assert row is not None

    def test_total_duration_positive(self, tmp_path):
        csv    = _make_csv(tmp_path)
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.total_duration_secs > 0.0

    def test_missing_csv_returns_failed(self, tmp_path):
        result = Orchestrator().run("/nonexistent/path.csv", "target", str(tmp_path / "db"))
        assert result.status == "failed"

    def test_missing_csv_error_message_not_none(self, tmp_path):
        result = Orchestrator().run("/nonexistent/path.csv", "target", str(tmp_path / "db"))
        assert result.error_message is not None

    def test_missing_csv_no_exception_raised(self, tmp_path):
        result = Orchestrator().run("/nonexistent/path.csv", "target", str(tmp_path / "db"))
        assert isinstance(result, AnalysisResult)

    def test_primary_metric_classification(self, tmp_path):
        csv    = _make_csv(tmp_path, task="classification")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.primary_metric == "macro_f1"

    def test_primary_metric_regression(self, tmp_path):
        csv    = _make_csv(tmp_path, task="regression")
        result = Orchestrator(config=self._cfg()).run(csv, "target", str(tmp_path / "db"))
        assert result.primary_metric == "rmse"

    def test_evaluation_attempts_in_db(self, tmp_path):
        csv    = _make_csv(tmp_path)
        db     = str(tmp_path / "db.sqlite")
        result = Orchestrator(config=self._cfg()).run(csv, "target", db)
        conn   = sqlite3.connect(db)
        rows   = conn.execute(
            "SELECT id FROM evaluation_attempts WHERE analysis_run_id = ?",
            (result.run_id,),
        ).fetchall()
        conn.close()
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# TestStubLLMClient
# ---------------------------------------------------------------------------

class TestStubLLMClient:
    def test_first_suggestion_valid(self):
        from src.evaluator import validate_strategy
        from src.llm_client import StubLLMClient

        s = StubLLMClient().suggest_strategy({}, "classification", set(), 1)
        assert s is not None
        assert validate_strategy(s) is True

    def test_no_duplicate_suggestions(self):
        from src.llm_client import StubLLMClient
        from src.memory import compute_strategy_signature

        client = StubLLMClient()
        tried: set[str] = set()

        for i in range(1, 10):
            s = client.suggest_strategy({}, "classification", tried, i)
            if s is None:
                break
            sig = compute_strategy_signature(s)
            assert sig not in tried
            tried.add(sig)

    def test_six_unique_candidates(self):
        from src.llm_client import StubLLMClient
        from src.memory import compute_strategy_signature

        client  = StubLLMClient()
        tried:  set[str] = set()
        seen:   list[dict] = []

        for i in range(1, 20):
            s = client.suggest_strategy({}, "classification", tried, i)
            if s is None:
                break
            tried.add(compute_strategy_signature(s))
            seen.append(s)

        assert len(seen) == 6

    def test_returns_none_when_all_tried(self):
        from src.llm_client import StubLLMClient, _CANDIDATES
        from src.memory import compute_strategy_signature

        tried  = {compute_strategy_signature(c) for c in _CANDIDATES}
        result = StubLLMClient().suggest_strategy({}, "classification", tried, 7)
        assert result is None


# ---------------------------------------------------------------------------
# TestSaveAnalysisReport
# ---------------------------------------------------------------------------

class TestSaveAnalysisReport:
    def _db_conn(self, tmp_path):
        from src.memory import init_db
        return init_db(str(tmp_path / "test.db"))

    def _run_id(self, conn):
        from src.memory import create_analysis_run, save_dataset
        did = save_dataset(conn, {
            "name":      "test.csv",
            "file_path": "/tmp/test_report.csv",
            "file_hash": "deadbeef1234",
        })
        return create_analysis_run(conn, did, "test request")

    def test_returns_positive_id(self, tmp_path):
        from src.memory import save_analysis_report
        conn = self._db_conn(tmp_path)
        rid  = self._run_id(conn)
        rid2 = save_analysis_report(conn, rid, None, {}, total_attempts=1)
        assert isinstance(rid2, int) and rid2 > 0

    def test_report_stored_in_db(self, tmp_path):
        from src.memory import save_analysis_report
        conn = self._db_conn(tmp_path)
        rid  = self._run_id(conn)
        save_analysis_report(
            conn, rid, None,
            {"final_test_primary_score": 0.85},
            total_attempts=2,
        )
        row = conn.execute(
            "SELECT final_test_primary_score, total_attempts FROM analysis_reports "
            "WHERE analysis_run_id = ?",
            (rid,),
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.85) < 1e-9
        assert row[1] == 2

    def test_duplicate_run_id_raises_integrity_error(self, tmp_path):
        from src.memory import save_analysis_report
        conn = self._db_conn(tmp_path)
        rid  = self._run_id(conn)
        save_analysis_report(conn, rid, None, {}, total_attempts=1)
        with pytest.raises(sqlite3.IntegrityError):
            save_analysis_report(conn, rid, None, {}, total_attempts=1)

    def test_total_duration_stored(self, tmp_path):
        from src.memory import save_analysis_report
        conn = self._db_conn(tmp_path)
        rid  = self._run_id(conn)
        save_analysis_report(
            conn, rid, None, {},
            total_duration_seconds=12.5, total_attempts=1,
        )
        row = conn.execute(
            "SELECT total_duration_seconds FROM analysis_reports WHERE analysis_run_id = ?",
            (rid,),
        ).fetchone()
        assert abs(row[0] - 12.5) < 1e-9

    def test_cv_primary_score_stored(self, tmp_path):
        from src.memory import save_analysis_report
        conn = self._db_conn(tmp_path)
        rid  = self._run_id(conn)
        save_analysis_report(
            conn, rid, None, {},
            cv_primary_score=0.72, total_attempts=1,
        )
        row = conn.execute(
            "SELECT cv_primary_score FROM analysis_reports WHERE analysis_run_id = ?",
            (rid,),
        ).fetchone()
        assert abs(row[0] - 0.72) < 1e-9
