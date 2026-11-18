"""
FR-08 — SQLite Tabanlı Deneyim Belleği birim testleri.

Her test geçici bir SQLite dosyası kullanır (tmp_path fixture);
testler birbirinden tamamen izole çalışır.
"""

import json
import sqlite3

import pytest

from src.memory import (
    build_profile_summary,
    compute_profile_hash,
    compute_strategy_signature,
    create_analysis_run,
    init_db,
    query_similar_strategies,
    safe_record_attempt,
    save_dataset,
    save_dataset_profile,
    save_evaluation_attempt,
    update_analysis_run,
    update_experience_memory,
    upsert_strategy,
)


# ---------------------------------------------------------------------------
# Ortak test sabitleri
# ---------------------------------------------------------------------------

STRATEGY_A = {
    "imputation_numeric":     "median",
    "imputation_categorical": "most_frequent",
    "scaling_method":         "standard",
    "encoding_method":        "onehot",
    "outlier_handling":       "none",
    "feature_selection":      "none",
}

STRATEGY_B = {
    "imputation_numeric":     "mean",
    "imputation_categorical": "most_frequent",
    "scaling_method":         "robust",
    "encoding_method":        "onehot",
    "outlier_handling":       "none",
    "feature_selection":      "none",
}

DATASET_INFO = {
    "name":      "test_dataset",
    "file_path": "data/raw/test_dataset.csv",
    "file_hash": "abc123def456" * 4,  # 48 karakter — temsili SHA-256
    "source":    "upload",
    "row_count": 1000,
    "column_count": 8,
}

# generate_data_profile() çıktısını temsil eden minimum örnek profil.
SAMPLE_PROFILE = {
    "dataset_info": {
        "rows":    1000,
        "columns": 9,
        "target":  "Survived",
    },
    "column_types": {
        "numeric":      ["Age", "Fare", "SibSp"],
        "categorical":  ["Sex", "Embarked"],
    },
    "missing_values": {
        "Age":      {"missing_count": 177, "missing_percent": 19.86},
        "Fare":     {"missing_count": 0,   "missing_percent": 0.0},
        "Sex":      {"missing_count": 0,   "missing_percent": 0.0},
        "Embarked": {"missing_count": 2,   "missing_percent": 0.22},
    },
    "distribution": {
        "Age":  {"mean": 29.7, "median": 28.0, "std": 14.5, "skewness": 0.39},
        "Fare": {"mean": 32.2, "median": 14.5, "std": 49.7, "skewness": 4.79},
    },
    "outliers": [],
}


# ---------------------------------------------------------------------------
# Yardımcı: test veritabanı
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Her test için izole bir SQLite dosyası ve açık bağlantı sağlar."""
    db_file = str(tmp_path / "test.db")
    conn = init_db(db_file)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# 1. Şema ve başlatma
# ---------------------------------------------------------------------------

class TestInitDb:
    def test_tables_exist(self, db):
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "users", "datasets", "dataset_profiles",
            "analysis_runs", "preprocessing_strategies",
            "evaluation_attempts", "experience_memory",
            "analysis_reports",
        }
        assert expected.issubset(tables)

    def test_view_exists(self, db):
        views = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            ).fetchall()
        }
        assert "v_strategy_performance" in views

    def test_system_user_created(self, db):
        row = db.execute(
            "SELECT id, email FROM users WHERE id = 1"
        ).fetchone()
        assert row is not None
        assert row["email"] == "system@local"

    def test_idempotent(self, tmp_path):
        """init_db iki kez çağrılsa da hata vermemeli."""
        db_file = str(tmp_path / "idem.db")
        conn1 = init_db(db_file)
        conn1.close()
        conn2 = init_db(db_file)
        conn2.close()


# ---------------------------------------------------------------------------
# 2. compute_strategy_signature
# ---------------------------------------------------------------------------

class TestComputeStrategySignature:
    def test_deterministic(self):
        sig1 = compute_strategy_signature(STRATEGY_A)
        sig2 = compute_strategy_signature(STRATEGY_A)
        assert sig1 == sig2

    def test_different_strategies_differ(self):
        sig_a = compute_strategy_signature(STRATEGY_A)
        sig_b = compute_strategy_signature(STRATEGY_B)
        assert sig_a != sig_b

    def test_returns_hex_string(self):
        sig = compute_strategy_signature(STRATEGY_A)
        int(sig, 16)  # geçerli hex ise hata vermez

    def test_extra_params_affect_signature(self):
        s1 = compute_strategy_signature(STRATEGY_A)
        s_with_extra = {**STRATEGY_A, "extra_params": {"knn_neighbors": 5}}
        s2 = compute_strategy_signature(s_with_extra)
        assert s1 != s2


# ---------------------------------------------------------------------------
# 3. compute_profile_hash
# ---------------------------------------------------------------------------

class TestComputeProfileHash:
    def test_deterministic(self):
        summary = {
            "numeric_column_count":     3,
            "categorical_column_count": 2,
            "total_missing_pct":        5.0,
            "problem_type":             "classification",
        }
        h1 = compute_profile_hash(summary)
        h2 = compute_profile_hash(summary)
        assert h1 == h2

    def test_similar_datasets_same_hash(self):
        """Demet boyutları sayesinde yakın profiller aynı hash'e düşer."""
        # num 3→bucket 1, num 4→bucket 1 (her ikisi 3//3=1, 4//3=1)
        s1 = {
            "numeric_column_count": 3,
            "categorical_column_count": 2,
            "total_missing_pct": 5.0,
            "problem_type": "classification",
        }
        s2 = {
            "numeric_column_count": 4,
            "categorical_column_count": 3,
            "total_missing_pct": 7.0,
            "problem_type": "classification",
        }
        assert compute_profile_hash(s1) == compute_profile_hash(s2)

    def test_different_problem_type_differs(self):
        s_clf = {
            "numeric_column_count": 3,
            "categorical_column_count": 2,
            "total_missing_pct": 5.0,
            "problem_type": "classification",
        }
        s_reg = {**s_clf, "problem_type": "regression"}
        assert compute_profile_hash(s_clf) != compute_profile_hash(s_reg)

    def test_very_different_missing_differs(self):
        s_low  = {"numeric_column_count": 3, "categorical_column_count": 2,
                  "total_missing_pct": 5.0,  "problem_type": "classification"}
        s_high = {"numeric_column_count": 3, "categorical_column_count": 2,
                  "total_missing_pct": 55.0, "problem_type": "classification"}
        assert compute_profile_hash(s_low) != compute_profile_hash(s_high)


# ---------------------------------------------------------------------------
# 4. build_profile_summary
# ---------------------------------------------------------------------------

class TestBuildProfileSummary:
    def test_numeric_and_categorical_counts(self):
        summary = build_profile_summary(SAMPLE_PROFILE, "classification")
        assert summary["numeric_column_count"] == 3
        assert summary["categorical_column_count"] == 2

    def test_total_missing_pct_calculated(self):
        summary = build_profile_summary(SAMPLE_PROFILE, "classification")
        # 4 sütun: 19.86 + 0 + 0 + 0.22 / 4 ≈ 5.02
        assert summary["total_missing_pct"] > 0
        assert summary["total_missing_pct"] < 25

    def test_problem_type_passed_through(self):
        summary = build_profile_summary(SAMPLE_PROFILE, "regression")
        assert summary["problem_type"] == "regression"


# ---------------------------------------------------------------------------
# 5. upsert_strategy
# ---------------------------------------------------------------------------

class TestUpsertStrategy:
    def test_creates_new_strategy(self, db):
        sid = upsert_strategy(db, STRATEGY_A)
        assert isinstance(sid, int)
        assert sid > 0

    def test_idempotent(self, db):
        sid1 = upsert_strategy(db, STRATEGY_A)
        sid2 = upsert_strategy(db, STRATEGY_A)
        assert sid1 == sid2

    def test_different_strategies_get_different_ids(self, db):
        sid_a = upsert_strategy(db, STRATEGY_A)
        sid_b = upsert_strategy(db, STRATEGY_B)
        assert sid_a != sid_b

    def test_strategy_stored_correctly(self, db):
        sid = upsert_strategy(db, STRATEGY_A)
        row = db.execute(
            "SELECT * FROM preprocessing_strategies WHERE id = ?", (sid,)
        ).fetchone()
        assert row["imputation_numeric"] == "median"
        assert row["scaling_method"] == "standard"


# ---------------------------------------------------------------------------
# 6. save_dataset — hash ile deduplication
# ---------------------------------------------------------------------------

class TestSaveDataset:
    def test_creates_dataset(self, db):
        did = save_dataset(db, DATASET_INFO)
        assert isinstance(did, int) and did > 0

    def test_hash_deduplication(self, db):
        """Aynı file_hash ile farklı ad kullanılsa da aynı id döner (AC 5)."""
        did1 = save_dataset(db, DATASET_INFO)
        duplicate = {**DATASET_INFO, "name": "different_name"}
        did2 = save_dataset(db, duplicate)
        assert did1 == did2

    def test_different_hashes_get_different_ids(self, db):
        info2 = {**DATASET_INFO, "file_hash": "x" * 48, "file_path": "other.csv"}
        did1 = save_dataset(db, DATASET_INFO)
        did2 = save_dataset(db, info2)
        assert did1 != did2


# ---------------------------------------------------------------------------
# 7. create_analysis_run / update_analysis_run
# ---------------------------------------------------------------------------

class TestAnalysisRun:
    def _make_run(self, db):
        did = save_dataset(db, DATASET_INFO)
        return create_analysis_run(db, did, "test request"), did

    def test_creates_run(self, db):
        run_id, _ = self._make_run(db)
        assert isinstance(run_id, int) and run_id > 0

    def test_initial_status_running(self, db):
        run_id, _ = self._make_run(db)
        row = db.execute(
            "SELECT status FROM analysis_runs WHERE id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "running"

    def test_update_to_completed(self, db):
        run_id, _ = self._make_run(db)
        update_analysis_run(db, run_id, "completed", attempt_count=3)
        row = db.execute(
            "SELECT status, attempt_count, finished_at FROM analysis_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert row["status"] == "completed"
        assert row["attempt_count"] == 3
        assert row["finished_at"] is not None

    def test_update_to_failed_with_message(self, db):
        run_id, _ = self._make_run(db)
        update_analysis_run(db, run_id, "failed", error_message="İşlem zaman aşımına uğradı")
        row = db.execute(
            "SELECT status, error_message FROM analysis_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert "zaman" in row["error_message"]


# ---------------------------------------------------------------------------
# 8. save_evaluation_attempt — başarılı ve başarısız denemeler
# ---------------------------------------------------------------------------

class TestSaveEvaluationAttempt:
    def _setup(self, db):
        did    = save_dataset(db, DATASET_INFO)
        run_id = create_analysis_run(db, did, "test")
        sid    = upsert_strategy(db, STRATEGY_A)
        return run_id, sid

    def test_saves_successful_attempt(self, db):
        run_id, sid = self._setup(db)
        attempt_data = {
            "attempt_number":    1,
            "model_type":        "logistic_regression",
            "primary_metric_name": "macro_f1",
            "cv_primary_score":  0.82,
            "status":            "success",
        }
        aid = save_evaluation_attempt(db, run_id, sid, attempt_data)
        assert aid > 0

    def test_saves_failed_attempt(self, db):
        """Başarısız denemeler de kaydedilmelidir (AC 1)."""
        run_id, sid = self._setup(db)
        attempt_data = {
            "attempt_number":    1,
            "model_type":        "logistic_regression",
            "primary_metric_name": "macro_f1",
            "cv_primary_score":  None,
            "status":            "failed",
            "error_message":     "Pipeline oluşturma hatası",
        }
        aid = save_evaluation_attempt(db, run_id, sid, attempt_data)
        row = db.execute(
            "SELECT status, error_message, cv_primary_score"
            " FROM evaluation_attempts WHERE id = ?",
            (aid,),
        ).fetchone()
        assert row["status"] == "failed"
        assert row["error_message"] == "Pipeline oluşturma hatası"
        assert row["cv_primary_score"] is None

    def test_cv_scores_detail_stored_as_json(self, db):
        run_id, sid = self._setup(db)
        detail = {"fold_scores": [0.80, 0.82, 0.78, 0.85, 0.83], "mean": 0.816, "std": 0.024}
        attempt_data = {
            "attempt_number":    1,
            "model_type":        "logistic_regression",
            "primary_metric_name": "macro_f1",
            "cv_primary_score":  0.816,
            "cv_scores_detail":  detail,
            "status":            "success",
        }
        aid = save_evaluation_attempt(db, run_id, sid, attempt_data)
        row = db.execute(
            "SELECT cv_scores_detail FROM evaluation_attempts WHERE id = ?",
            (aid,),
        ).fetchone()
        stored = json.loads(row["cv_scores_detail"])
        assert stored["mean"] == pytest.approx(0.816)

    def test_attempt_number_uniqueness(self, db):
        """Aynı analiz içinde aynı deneme numarası iki kez eklenemez."""
        run_id, sid = self._setup(db)
        sid2 = upsert_strategy(db, STRATEGY_B)
        base = {
            "attempt_number": 1,
            "model_type":     "logistic_regression",
            "primary_metric_name": "macro_f1",
            "status":         "success",
        }
        save_evaluation_attempt(db, run_id, sid, base)
        with pytest.raises(sqlite3.IntegrityError):
            save_evaluation_attempt(db, run_id, sid2, base)

    def test_strategy_uniqueness_per_run(self, db):
        """Aynı analiz içinde aynı strateji iki kez denenemez."""
        run_id, sid = self._setup(db)
        a1 = {"attempt_number": 1, "model_type": "logistic_regression",
               "primary_metric_name": "macro_f1", "status": "success"}
        a2 = {"attempt_number": 2, "model_type": "logistic_regression",
               "primary_metric_name": "macro_f1", "status": "success"}
        save_evaluation_attempt(db, run_id, sid, a1)
        with pytest.raises(sqlite3.IntegrityError):
            save_evaluation_attempt(db, run_id, sid, a2)


# ---------------------------------------------------------------------------
# 9. update_experience_memory — UPSERT mantığı
# ---------------------------------------------------------------------------

class TestUpdateExperienceMemory:
    def _setup(self, db):
        did    = save_dataset(db, DATASET_INFO)
        run_id = create_analysis_run(db, did, "test")
        sid    = upsert_strategy(db, STRATEGY_A)
        p_hash = "testhash_clf"
        return did, run_id, sid, p_hash

    def test_creates_new_entry(self, db):
        did, run_id, sid, p_hash = self._setup(db)
        update_experience_memory(
            db, p_hash, sid, "classification", 0.80, 1.5, True, did, run_id
        )
        row = db.execute(
            "SELECT * FROM experience_memory"
            " WHERE profile_hash = ? AND strategy_id = ?",
            (p_hash, sid),
        ).fetchone()
        assert row is not None
        assert row["cv_score_avg"] == pytest.approx(0.80)
        assert row["cv_score_best"] == pytest.approx(0.80)
        assert row["attempt_count"] == 1
        assert row["success_count"] == 1

    def test_upsert_updates_avg_score(self, db):
        did, run_id, sid, p_hash = self._setup(db)
        update_experience_memory(db, p_hash, sid, "classification", 0.80, 1.0, True, did, run_id)
        update_experience_memory(db, p_hash, sid, "classification", 0.90, 1.0, True, did, run_id)

        row = db.execute(
            "SELECT cv_score_avg, cv_score_best, attempt_count, success_count"
            " FROM experience_memory WHERE profile_hash = ? AND strategy_id = ?",
            (p_hash, sid),
        ).fetchone()
        assert row["cv_score_avg"] == pytest.approx(0.85)
        assert row["cv_score_best"] == pytest.approx(0.90)
        assert row["attempt_count"] == 2
        assert row["success_count"] == 2

    def test_failed_attempt_increments_count_not_success(self, db):
        did, run_id, sid, p_hash = self._setup(db)
        update_experience_memory(db, p_hash, sid, "classification", 0.80, 1.0, True,  did, run_id)
        update_experience_memory(db, p_hash, sid, "classification", 0.60, 1.0, False, did, run_id)

        row = db.execute(
            "SELECT attempt_count, success_count FROM experience_memory"
            " WHERE profile_hash = ? AND strategy_id = ?",
            (p_hash, sid),
        ).fetchone()
        assert row["attempt_count"] == 2
        assert row["success_count"] == 1

    def test_different_problem_types_separate_rows(self, db):
        did, run_id, sid, p_hash = self._setup(db)
        update_experience_memory(db, p_hash, sid, "classification", 0.80, None, True, did, run_id)
        update_experience_memory(db, p_hash, sid, "regression",     0.75, None, True, did, run_id)

        count = db.execute(
            "SELECT COUNT(*) FROM experience_memory WHERE profile_hash = ? AND strategy_id = ?",
            (p_hash, sid),
        ).fetchone()[0]
        assert count == 2

    def test_avg_duration_updated(self, db):
        did, run_id, sid, p_hash = self._setup(db)
        update_experience_memory(db, p_hash, sid, "classification", 0.80, 2.0, True, did, run_id)
        update_experience_memory(db, p_hash, sid, "classification", 0.80, 4.0, True, did, run_id)

        row = db.execute(
            "SELECT avg_duration_seconds FROM experience_memory"
            " WHERE profile_hash = ? AND strategy_id = ?",
            (p_hash, sid),
        ).fetchone()
        assert row["avg_duration_seconds"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 10. query_similar_strategies — sorgulama ve dışlama
# ---------------------------------------------------------------------------

class TestQuerySimilarStrategies:
    def _populate(self, db):
        """İki farklı veri seti, iki farklı strateji, aynı profil hash'i."""
        info1 = {**DATASET_INFO}
        info2 = {
            **DATASET_INFO,
            "file_hash": "b" * 48,
            "file_path": "data/raw/other.csv",
            "name": "other_dataset",
        }
        did1 = save_dataset(db, info1)
        did2 = save_dataset(db, info2)

        run1 = create_analysis_run(db, did1, "run1")
        run2 = create_analysis_run(db, did2, "run2")

        sid_a = upsert_strategy(db, STRATEGY_A)
        sid_b = upsert_strategy(db, STRATEGY_B)

        p_hash = "shared_profile_hash"

        update_experience_memory(db, p_hash, sid_a, "classification", 0.85, 1.0, True,  did1, run1)
        update_experience_memory(db, p_hash, sid_b, "classification", 0.78, 1.0, False, did2, run2)

        return did1, did2, p_hash, sid_a, sid_b

    def test_returns_strategies(self, db):
        did1, did2, p_hash, sid_a, sid_b = self._populate(db)
        results = query_similar_strategies(db, p_hash, "classification")
        assert len(results) == 2

    def test_ordered_by_cv_score_desc(self, db):
        did1, did2, p_hash, sid_a, sid_b = self._populate(db)
        results = query_similar_strategies(db, p_hash, "classification")
        scores = [r["cv_score_avg"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_excludes_current_dataset(self, db):
        """Mevcut veri setine ait kayıtlar sorgudan çıkarılmalı (AC 4)."""
        did1, did2, p_hash, sid_a, sid_b = self._populate(db)
        # did1 dışlanırsa yalnızca sid_b (did2'ye ait) görünmeli
        results = query_similar_strategies(
            db, p_hash, "classification", exclude_dataset_id=did1
        )
        returned_ids = {r["strategy_id"] for r in results}
        assert sid_a not in returned_ids
        assert sid_b in returned_ids

    def test_no_results_for_unknown_hash(self, db):
        results = query_similar_strategies(db, "bilinmeyen_hash", "classification")
        assert results == []

    def test_limit_respected(self, db):
        did1, did2, p_hash, sid_a, sid_b = self._populate(db)
        results = query_similar_strategies(db, p_hash, "classification", limit=1)
        assert len(results) == 1

    def test_result_contains_strategy_fields(self, db):
        did1, did2, p_hash, sid_a, sid_b = self._populate(db)
        results = query_similar_strategies(db, p_hash, "classification")
        r = results[0]
        for field in (
            "strategy_id", "cv_score_avg", "cv_score_best",
            "attempt_count", "success_count", "success_rate",
            "imputation_numeric", "scaling_method",
        ):
            assert field in r, f"'{field}' alanı eksik"


# ---------------------------------------------------------------------------
# 11. safe_record_attempt — hata toleransı
# ---------------------------------------------------------------------------

class TestSafeRecordAttempt:
    def test_returns_attempt_id_on_success(self, db):
        did    = save_dataset(db, DATASET_INFO)
        run_id = create_analysis_run(db, did, "test")
        sid    = upsert_strategy(db, STRATEGY_A)
        p_hash = "safe_test_hash"

        result = safe_record_attempt(
            db, run_id, sid, p_hash, "classification",
            {
                "attempt_number": 1,
                "model_type":     "logistic_regression",
                "primary_metric_name": "macro_f1",
                "cv_primary_score": 0.81,
                "duration_seconds": 2.3,
                "status": "success",
            },
            dataset_id=did,
        )
        assert result is not None and result > 0

    def test_does_not_raise_on_db_error(self, tmp_path):
        """
        Bozuk bağlantı vesilesiyle ortaya çıkan DB hatası
        exception fırlatmamalı; None dönmeli (AC 6).
        """
        db_file = str(tmp_path / "err.db")
        conn = init_db(db_file)
        conn.close()  # Kapalı bağlantıya yazma → hata

        result = safe_record_attempt(
            conn, 999, 999, "hash", "classification",
            {"attempt_number": 1, "status": "success"},
        )
        assert result is None

    def test_experience_memory_updated_after_save(self, db):
        did    = save_dataset(db, DATASET_INFO)
        run_id = create_analysis_run(db, did, "test")
        sid    = upsert_strategy(db, STRATEGY_A)
        p_hash = "exp_mem_test"

        safe_record_attempt(
            db, run_id, sid, p_hash, "classification",
            {
                "attempt_number": 1,
                "model_type":     "logistic_regression",
                "primary_metric_name": "macro_f1",
                "cv_primary_score": 0.75,
                "status": "success",
            },
            dataset_id=did,
        )

        row = db.execute(
            "SELECT cv_score_avg FROM experience_memory"
            " WHERE profile_hash = ? AND strategy_id = ?",
            (p_hash, sid),
        ).fetchone()
        assert row is not None
        assert row["cv_score_avg"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 12. save_dataset_profile
# ---------------------------------------------------------------------------

class TestSaveDatasetProfile:
    def test_saves_profile(self, db):
        did     = save_dataset(db, DATASET_INFO)
        summary = build_profile_summary(SAMPLE_PROFILE, "classification")
        p_hash  = compute_profile_hash(summary)

        pid = save_dataset_profile(db, did, SAMPLE_PROFILE, summary, p_hash)
        assert pid > 0

        row = db.execute(
            "SELECT numeric_column_count, categorical_column_count,"
            " problem_type, profile_hash"
            " FROM dataset_profiles WHERE id = ?",
            (pid,),
        ).fetchone()
        assert row["numeric_column_count"] == 3
        assert row["categorical_column_count"] == 2
        assert row["problem_type"] == "classification"
        assert row["profile_hash"] == p_hash

    def test_idempotent_for_same_dataset(self, db):
        did     = save_dataset(db, DATASET_INFO)
        summary = build_profile_summary(SAMPLE_PROFILE, "classification")
        p_hash  = compute_profile_hash(summary)

        pid1 = save_dataset_profile(db, did, SAMPLE_PROFILE, summary, p_hash)
        pid2 = save_dataset_profile(db, did, SAMPLE_PROFILE, summary, p_hash)
        assert pid1 == pid2
