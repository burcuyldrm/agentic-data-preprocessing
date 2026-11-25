"""
FR-07 — Geri Bildirim Döngüsü ve Özdenetim Orkestrasyonu.

Dışa açık sınıf: Orchestrator
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.core.hashing import compute_profile_hash, compute_strategy_signature
from src.datasets.split     import split_dataset
from src.profiling.detector import detect_column_types
from src.profiling.loader   import load_dataset
from src.profiling.profiler import generate_data_profile
from src.datasets.repository import build_profile_summary, save_dataset, save_dataset_profile
from src.evaluation.metrics import (
    build_preprocessor_from_strategy,
    evaluate_on_test,
    evaluate_strategy,
    validate_strategy,
)
from src.evaluation.repository import save_evaluation_attempt
from src.experience_memory.store import query_similar_strategies, safe_record_attempt
from src.llm_client import StubLLMClient
from src.preprocessing.pipeline import build_baseline_preprocessor
from src.preprocessing.strategy import select_preprocessing_strategy
from src.memory import create_analysis_run, save_analysis_report, update_analysis_run
from src.orchestration.config import OrchestratorConfig
from src.orchestration.models import AnalysisResult, AttemptRecord
from src.orchestration.utils import _file_hash, _is_better
from src.persistence.db import init_db
from src.preprocessing.repository import upsert_strategy
from src.task_detection import detect_task_type

logger = logging.getLogger(__name__)

# Mod A sabit strateji. Mod B türetilmiş stratejiyle aynı anahtar kümesini kullanır.
_BASELINE_STRATEGY: dict = {
    "imputation_numeric":    "median",
    "imputation_categorical": "most_frequent",
    "scaling_method":        "standard",
    "encoding_method":       "onehot",
    "outlier_handling":      "none",
    "feature_selection":     "none",
}


@dataclass
class _RunContext:
    """
    run() boyunca taşınan çalışma durumu.

    Private metotlar arası state geçişini sağlar; döngü değişkenlerinin
    yanlışlıkla sıfırlanmasını engeller.
    """
    conn:    sqlite3.Connection
    cfg:     OrchestratorConfig
    t_start: float
    # _load_and_register
    run_id:     Optional[int] = None
    dataset_id: Optional[int] = None
    task_type:  str           = ""
    # _split_and_profile
    X_train:         Any  = None
    X_test:          Any  = None
    y_train:         Any  = None
    y_test:          Any  = None
    numeric_cols:    list = field(default_factory=list)
    cat_cols:        list = field(default_factory=list)
    profile_summary: dict = field(default_factory=dict)
    profile_hash:    str  = ""
    # _query_memory
    memory_strategies: list = field(default_factory=list)
    # _run_feedback_loop
    attempts:            list          = field(default_factory=list)
    best_score:          Optional[float] = None
    best_strategy:       Optional[dict]  = None
    best_strategy_id:    Optional[int]   = None
    best_primary_metric: Optional[str]   = None
    # _run_feedback_loop / _run_single_attempt
    llm_call_count: int = 0
    # _evaluate_final
    final_scores: Optional[dict] = None


class Orchestrator:
    """
    Geri bildirim döngüsü orkestrasyonu (FR-07, Mod C/D).

    Akış:
      1. Veri setini böl (split_dataset) — veri sızıntısını önler.
      2. Eğitim setini profille (generate_data_profile).
      3. Bellek destekli modda (C/D) benzer stratejileri sorgula.
      4. Döngü: strateji seç → önişleme yap → CV değerlendir →
                durdurma koşulunu kontrol et → belleği güncelle.
      5. En iyi stratejiyle test setini değerlendir (bir kez).
      6. Raporu kaydet.
    """

    def __init__(
        self,
        llm_client=None,
        config: OrchestratorConfig | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.llm    = llm_client or StubLLMClient()

    def run(
        self,
        csv_path:     str,
        target:       str,
        db_path:      str,
        user_request: str = (
            "Veri setini analiz et ve en iyi önişleme stratejisini bul."
        ),
        user_id:      int = 1,
    ) -> AnalysisResult:
        t_start = time.time()
        conn    = init_db(db_path)
        ctx     = _RunContext(conn=conn, cfg=self.config, t_start=t_start)

        try:
            df = self._load_and_register(ctx, csv_path, target, user_request, user_id)
            self._split_and_profile(ctx, target, df)
            if ctx.cfg.experiment_mode in ("A", "B"):
                self._run_mode_ab(ctx)
            else:
                self._query_memory(ctx)
                self._run_feedback_loop(ctx)
            self._evaluate_final(ctx)
            return self._save_and_return(ctx)
        except Exception as exc:
            return self._handle_error(ctx, exc)

    # ----------------------------------------------------------------
    # Adım 1 — Veri yükleme ve kayıt
    # ----------------------------------------------------------------

    def _load_and_register(
        self,
        ctx:          _RunContext,
        csv_path:     str,
        target:       str,
        user_request: str,
        user_id:      int,
    ) -> Any:
        df        = load_dataset(csv_path)
        task_type = detect_task_type(df, target)
        file_hash = _file_hash(csv_path)

        dataset_id = save_dataset(ctx.conn, {
            "name":         csv_path.split("/")[-1],
            "file_path":    csv_path,
            "file_hash":    file_hash,
            "row_count":    len(df),
            "column_count": len(df.columns),
        })
        run_id = create_analysis_run(
            ctx.conn,
            dataset_id      = dataset_id,
            user_request    = user_request,
            experiment_mode = ctx.cfg.experiment_mode,
            random_seed     = ctx.cfg.random_seed,
            max_attempts    = ctx.cfg.max_attempts,
            user_id         = user_id,
        )

        ctx.dataset_id = dataset_id
        ctx.run_id     = run_id
        ctx.task_type  = task_type
        return df

    # ----------------------------------------------------------------
    # Adım 2 — Bölme ve profilleme
    # ----------------------------------------------------------------

    def _split_and_profile(self, ctx: _RunContext, target: str, df: Any) -> None:
        X_train, X_test, y_train, y_test = split_dataset(
            df, target,
            task_type    = ctx.task_type,
            random_state = ctx.cfg.random_seed,
        )

        df_train = X_train.copy()
        df_train[target] = y_train.values

        profile   = generate_data_profile(df_train, target)
        col_types = detect_column_types(df_train, target)

        profile_summary = build_profile_summary(profile, ctx.task_type)
        profile_hash    = compute_profile_hash(profile_summary)

        save_dataset_profile(ctx.conn, ctx.dataset_id, profile, profile_summary, profile_hash)

        ctx.X_train         = X_train
        ctx.X_test          = X_test
        ctx.y_train         = y_train
        ctx.y_test          = y_test
        ctx.numeric_cols    = col_types["numeric"]
        ctx.cat_cols        = col_types["categorical"]
        ctx.profile_summary = profile_summary
        ctx.profile_hash    = profile_hash

    # ----------------------------------------------------------------
    # Adım 3 — Bellek sorgusu
    # ----------------------------------------------------------------

    def _query_memory(self, ctx: _RunContext) -> None:
        # Yalnızca Mod D geçmiş deneyim belleğini önyükler.
        # Mod C LLM döngüsü kullanır ama bellek önyüklemesi yapmaz.
        if ctx.cfg.experiment_mode == "D":
            rows = query_similar_strategies(
                ctx.conn, ctx.profile_hash, ctx.task_type,
                exclude_dataset_id=ctx.dataset_id,
                limit=3,
            )
            ctx.memory_strategies = [dict(r) for r in rows]

    # ----------------------------------------------------------------
    # Mod A/B — Tek deneme ortak mekanizması
    # ----------------------------------------------------------------

    def _run_mode_ab(self, ctx: _RunContext) -> None:
        """
        Mod A: sabit baseline preprocessor, LLM yok.
        Mod B: profil tabanlı deterministik strateji, LLM yok.

        Her ikisi de tek deneme çalıştırır ve yalnızca evaluation_attempts'e
        yazar; experience_memory izolasyonu korunur.

        NOT — Deneme bütçesi farkı: A/B sabit 1 deneme; C/D max_attempts'e
        kadar döngü çalıştırır. Kontrollü deneylerde bu fark bir deneysel
        değişken olarak ele alınmalıdır.
        """
        if ctx.cfg.experiment_mode == "A":
            strategy    = dict(_BASELINE_STRATEGY)
            preprocessor = build_baseline_preprocessor(ctx.numeric_cols, ctx.cat_cols)
        else:
            strategy    = self._derive_strategy_from_profile(ctx)
            preprocessor = build_preprocessor_from_strategy(
                strategy, ctx.numeric_cols, ctx.cat_cols
            )
        self._run_single_attempt(ctx, strategy, preprocessor)

    def _run_single_attempt(
        self,
        ctx: _RunContext,
        strategy: dict,
        preprocessor,
    ) -> None:
        """
        Tek deneme çalıştırır. Yalnızca evaluation_attempts'e yazar;
        experience_memory'e yazmaz (Mod A/B bellek izolasyonu).
        """
        pmetric = "macro_f1" if ctx.task_type == "classification" else "rmse"
        mtype   = (
            "logistic_regression" if ctx.task_type == "classification"
            else "ridge_regression"
        )
        strategy_id = upsert_strategy(ctx.conn, strategy)
        t_attempt   = time.time()

        try:
            cv_result = evaluate_strategy(
                ctx.X_train, ctx.y_train, preprocessor, ctx.task_type,
                n_splits    = ctx.cfg.n_cv_splits,
                random_seed = ctx.cfg.random_seed,
            )
            cv_score = cv_result["cv_primary_score"]
            duration = cv_result["duration_seconds"]
            pmetric  = cv_result["primary_metric_name"]

            record = AttemptRecord(
                attempt_number     = 1,
                strategy           = strategy,
                strategy_id        = strategy_id,
                cv_primary_score   = cv_score,
                cv_secondary_score = cv_result.get("cv_secondary_score"),
                primary_metric     = pmetric,
                duration_seconds   = duration,
                status             = "success",
            )

            save_evaluation_attempt(
                ctx.conn, ctx.run_id, strategy_id,
                {
                    "attempt_number":      1,
                    "model_type":          cv_result.get("model_type", mtype),
                    "primary_metric_name": pmetric,
                    "cv_primary_score":    cv_score,
                    "cv_secondary_score":  cv_result.get("cv_secondary_score"),
                    "cv_scores_detail":    cv_result.get("cv_scores_detail"),
                    "duration_seconds":    duration,
                    "llm_call_count":      0,
                    "status":              "success",
                },
            )

            ctx.attempts            = [record]
            ctx.best_score          = cv_score
            ctx.best_strategy       = strategy
            ctx.best_strategy_id    = strategy_id
            ctx.best_primary_metric = pmetric
            update_analysis_run(ctx.conn, ctx.run_id, "running", attempt_count=1)

        except Exception as exc:
            duration = time.time() - t_attempt
            logger.warning("Tek deneme değerlendirme hatası (mod A/B): %s", exc)
            record = AttemptRecord(
                attempt_number     = 1,
                strategy           = strategy,
                strategy_id        = strategy_id,
                cv_primary_score   = None,
                cv_secondary_score = None,
                primary_metric     = pmetric,
                duration_seconds   = duration,
                status             = "failed",
                error_message      = str(exc),
            )
            save_evaluation_attempt(
                ctx.conn, ctx.run_id, strategy_id,
                {
                    "attempt_number":      1,
                    "model_type":          mtype,
                    "primary_metric_name": pmetric,
                    "cv_primary_score":    None,
                    "duration_seconds":    duration,
                    "llm_call_count":      0,
                    "status":              "failed",
                    "error_message":       str(exc),
                },
            )
            ctx.attempts = [record]
            update_analysis_run(ctx.conn, ctx.run_id, "running", attempt_count=1)

    def _derive_strategy_from_profile(self, ctx: _RunContext) -> dict:
        """
        Eğitim verisinin çarpıklık dağılımına göre ön işleme stratejisi türetir.

        Çoğunluk oyu kuralı:
          - Sütunların yarısından fazlası |skewness| >= 1 → scaling="robust"
          - Eşitlik veya standart çoğunluğu → scaling="standard" (muhafazakâr)
          - imputation: "median" >= "mean" sayısı → "median" (eşitlikte median)
        Sayısal sütun yoksa veya türetme başarısız olursa baseline'a düşer.
        """
        if not ctx.numeric_cols:
            return dict(_BASELINE_STRATEGY)
        try:
            per_col = select_preprocessing_strategy(ctx.X_train, ctx.numeric_cols)
        except Exception as exc:
            logger.warning("Strateji türetme hatası, baseline'a düşüldü: %s", exc)
            return dict(_BASELINE_STRATEGY)
        if not per_col:
            return dict(_BASELINE_STRATEGY)

        scalers  = [v["scaler"]  for v in per_col.values()]
        imputers = [v["imputer"] for v in per_col.values()]

        scaling    = "robust"  if scalers.count("robust")  > scalers.count("standard") else "standard"
        imputation = "median"  if imputers.count("median") >= imputers.count("mean")    else "mean"

        return {
            "imputation_numeric":    imputation,
            "imputation_categorical": "most_frequent",
            "scaling_method":        scaling,
            "encoding_method":       "onehot",
            "outlier_handling":      "none",
            "feature_selection":     "none",
        }

    # ----------------------------------------------------------------
    # Adım 4 — Geri bildirim döngüsü
    # ----------------------------------------------------------------

    def _run_feedback_loop(self, ctx: _RunContext) -> None:
        attempts:            list[AttemptRecord] = []
        tried_signatures:    set[str]            = set()
        best_score:          Optional[float]     = None
        best_strategy:       Optional[dict]      = None
        best_strategy_id:    Optional[int]       = None
        best_primary_metric: Optional[str]       = None
        consecutive_fails:   int                 = 0

        mem_iter = iter(ctx.memory_strategies)

        for attempt_num in range(1, ctx.cfg.max_attempts + 1):
            if time.time() - ctx.t_start > ctx.cfg.time_limit_seconds:
                logger.info("Zaman sınırına ulaşıldı (attempt %d).", attempt_num)
                break

            strategy = self._select_strategy(ctx, mem_iter, tried_signatures, attempt_num)

            if strategy is None:
                logger.info("LLM strateji önerisi tükendi (attempt %d).", attempt_num)
                break

            sig = compute_strategy_signature(strategy)
            if sig in tried_signatures:
                continue
            tried_signatures.add(sig)

            if not validate_strategy(strategy):
                record = AttemptRecord(
                    attempt_number     = attempt_num,
                    strategy           = strategy,
                    strategy_id        = 0,
                    cv_primary_score   = None,
                    cv_secondary_score = None,
                    primary_metric     = "macro_f1" if ctx.task_type == "classification" else "rmse",
                    duration_seconds   = 0.0,
                    status             = "invalid",
                    error_message      = "Geçersiz strateji parametreleri.",
                )
                attempts.append(record)
                consecutive_fails += 1
                if consecutive_fails >= ctx.cfg.max_consecutive_failures:
                    break
                continue

            strategy_id = upsert_strategy(ctx.conn, strategy)

            t_attempt = time.time()
            try:
                preprocessor = build_preprocessor_from_strategy(
                    strategy, ctx.numeric_cols, ctx.cat_cols
                )
                cv_result = evaluate_strategy(
                    ctx.X_train, ctx.y_train, preprocessor, ctx.task_type,
                    n_splits    = ctx.cfg.n_cv_splits,
                    random_seed = ctx.cfg.random_seed,
                )
                cv_score = cv_result["cv_primary_score"]
                duration = cv_result["duration_seconds"]
                pmetric  = cv_result["primary_metric_name"]

                record = AttemptRecord(
                    attempt_number     = attempt_num,
                    strategy           = strategy,
                    strategy_id        = strategy_id,
                    cv_primary_score   = cv_score,
                    cv_secondary_score = cv_result.get("cv_secondary_score"),
                    primary_metric     = pmetric,
                    duration_seconds   = duration,
                    status             = "success",
                )
                consecutive_fails = 0

                if best_score is None or _is_better(cv_score, best_score, ctx.task_type):
                    best_score          = cv_score
                    best_strategy       = strategy
                    best_strategy_id    = strategy_id
                    best_primary_metric = pmetric

                safe_record_attempt(
                    ctx.conn, ctx.run_id, strategy_id, ctx.profile_hash, ctx.task_type,
                    {
                        "attempt_number":      attempt_num,
                        "model_type":          cv_result.get("model_type", "logistic_regression"),
                        "primary_metric_name": pmetric,
                        "cv_primary_score":    cv_score,
                        "cv_secondary_score":  cv_result.get("cv_secondary_score"),
                        "cv_scores_detail":    cv_result.get("cv_scores_detail"),
                        "duration_seconds":    duration,
                        "llm_call_count":      0,
                        "status":              "success",
                    },
                    dataset_id=ctx.dataset_id,
                )

            except Exception as exc:
                duration = time.time() - t_attempt
                logger.warning("Değerlendirme hatası (attempt %d): %s", attempt_num, exc)
                pmetric = "macro_f1" if ctx.task_type == "classification" else "rmse"
                mtype   = "logistic_regression" if ctx.task_type == "classification" else "ridge_regression"
                record = AttemptRecord(
                    attempt_number     = attempt_num,
                    strategy           = strategy,
                    strategy_id        = strategy_id,
                    cv_primary_score   = None,
                    cv_secondary_score = None,
                    primary_metric     = pmetric,
                    duration_seconds   = duration,
                    status             = "failed",
                    error_message      = str(exc),
                )
                consecutive_fails += 1
                safe_record_attempt(
                    ctx.conn, ctx.run_id, strategy_id, ctx.profile_hash, ctx.task_type,
                    {
                        "attempt_number":      attempt_num,
                        "model_type":          mtype,
                        "primary_metric_name": pmetric,
                        "cv_primary_score":    None,
                        "duration_seconds":    duration,
                        "llm_call_count":      0,
                        "status":              "failed",
                        "error_message":       str(exc),
                    },
                    dataset_id=ctx.dataset_id,
                )
                if consecutive_fails >= ctx.cfg.max_consecutive_failures:
                    break

            attempts.append(record)
            update_analysis_run(ctx.conn, ctx.run_id, "running", attempt_count=attempt_num)

        ctx.attempts            = attempts
        ctx.best_score          = best_score
        ctx.best_strategy       = best_strategy
        ctx.best_strategy_id    = best_strategy_id
        ctx.best_primary_metric = best_primary_metric

    def _select_strategy(
        self,
        ctx:              _RunContext,
        mem_iter:         Any,
        tried_signatures: set[str],
        attempt_num:      int,
    ) -> Optional[dict]:
        try:
            mem_row = next(mem_iter)
            return {
                "imputation_numeric":     mem_row["imputation_numeric"],
                "imputation_categorical": mem_row["imputation_categorical"],
                "scaling_method":         mem_row["scaling_method"],
                "encoding_method":        mem_row["encoding_method"],
                "outlier_handling":       mem_row.get("outlier_handling", "none"),
                "feature_selection":      mem_row.get("feature_selection", "none"),
            }
        except StopIteration:
            ctx.llm_call_count += 1
            return self.llm.suggest_strategy(
                ctx.profile_summary, ctx.task_type, tried_signatures, attempt_num
            )

    # ----------------------------------------------------------------
    # Adım 5 — Nihai test değerlendirmesi
    # ----------------------------------------------------------------

    def _evaluate_final(self, ctx: _RunContext) -> None:
        if ctx.best_strategy is None:
            return
        try:
            preprocessor = build_preprocessor_from_strategy(
                ctx.best_strategy, ctx.numeric_cols, ctx.cat_cols
            )
            ctx.final_scores = evaluate_on_test(
                ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test,
                preprocessor, ctx.task_type,
                random_seed=ctx.cfg.random_seed,
            )
        except Exception as exc:
            logger.error("Test değerlendirme hatası: %s", exc)

    # ----------------------------------------------------------------
    # Adım 6 — Rapor kaydı ve dönüş
    # ----------------------------------------------------------------

    def _save_and_return(self, ctx: _RunContext) -> AnalysisResult:
        total_duration = time.time() - ctx.t_start

        if ctx.best_strategy is not None:
            update_analysis_run(
                ctx.conn, ctx.run_id, "completed", attempt_count=len(ctx.attempts)
            )
            save_analysis_report(
                ctx.conn,
                run_id                 = ctx.run_id,
                best_strategy_id       = ctx.best_strategy_id,
                final_scores           = ctx.final_scores or {},
                cv_primary_score       = ctx.best_score,
                total_attempts         = len(ctx.attempts),
                total_duration_seconds = total_duration,
                total_llm_calls        = ctx.llm_call_count,
            )
            result_status = "completed"
        else:
            update_analysis_run(
                ctx.conn, ctx.run_id, "failed",
                attempt_count=len(ctx.attempts),
                error_message="Geçerli strateji bulunamadı.",
            )
            result_status = "failed"

        ctx.conn.close()
        return AnalysisResult(
            status              = result_status,
            task_type           = ctx.task_type,
            best_strategy       = ctx.best_strategy,
            best_cv_score       = ctx.best_score,
            primary_metric      = ctx.best_primary_metric,
            final_test_scores   = ctx.final_scores,
            attempts            = ctx.attempts,
            total_duration_secs = total_duration,
            run_id              = ctx.run_id,
            total_llm_calls     = ctx.llm_call_count,
        )

    # ----------------------------------------------------------------
    # Hata işleyici
    # ----------------------------------------------------------------

    def _handle_error(self, ctx: _RunContext, exc: Exception) -> AnalysisResult:
        logger.error("Orkestrasyon hatası: %s", exc, exc_info=True)
        if ctx.run_id is not None:
            try:
                update_analysis_run(ctx.conn, ctx.run_id, "failed", error_message=str(exc))
            except Exception:
                pass
        try:
            ctx.conn.close()
        except Exception:
            pass
        return AnalysisResult(
            status              = "failed",
            task_type           = "",
            best_strategy       = None,
            best_cv_score       = None,
            primary_metric      = None,
            final_test_scores   = None,
            attempts            = [],
            total_duration_secs = time.time() - ctx.t_start,
            run_id              = ctx.run_id,
            error_message       = str(exc),
            total_llm_calls     = ctx.llm_call_count,
        )
