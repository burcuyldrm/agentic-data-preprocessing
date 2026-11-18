"""
FR-07 — Geri Bildirim Döngüsü ve Özdenetim Orkestrasyonu.

Dışa açık sınıf ve yardımcılar:
  OrchestratorConfig — Çalıştırma parametreleri.
  AttemptRecord      — Tek deneme sonucu.
  AnalysisResult     — Orkestrasyonun nihai çıktısı.
  Orchestrator       — Ana geri bildirim döngüsü.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    max_attempts:             int   = 5
    max_consecutive_failures: int   = 2
    time_limit_seconds:       float = 600.0
    random_seed:              int   = 42
    experiment_mode:          str   = "C"
    n_cv_splits:              int   = 5


@dataclass
class AttemptRecord:
    attempt_number:     int
    strategy:           dict
    strategy_id:        int
    cv_primary_score:   Optional[float]
    cv_secondary_score: Optional[float]
    primary_metric:     str
    duration_seconds:   float
    status:             str            # 'success' | 'failed' | 'invalid'
    error_message:      Optional[str] = None


@dataclass
class AnalysisResult:
    status:              str           # 'completed' | 'failed'
    task_type:           str
    best_strategy:       Optional[dict]
    best_cv_score:       Optional[float]
    primary_metric:      Optional[str]
    final_test_scores:   Optional[dict]
    attempts:            list[AttemptRecord] = field(default_factory=list)
    total_duration_secs: float = 0.0
    run_id:              Optional[int] = None
    error_message:       Optional[str] = None


class Orchestrator:
    """
    Geri bildirim döngüsü orkestrasyonu (FR-07, Mod C/D).

    Akış:
      1. Veri setini böl (split_dataset) — veri sızıntısını önler.
      2. Eğitim setini profille (generate_data_profile).
      3. Bellek destekli modda (D) benzer stratejileri sorgula.
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
        from src.llm_client import StubLLMClient

        self.config = config or OrchestratorConfig()
        self.llm    = llm_client or StubLLMClient()

    def run(
        self,
        csv_path: str,
        target: str,
        db_path: str,
        user_request: str = (
            "Veri setini analiz et ve en iyi önişleme stratejisini bul."
        ),
        user_id: int = 1,
    ) -> AnalysisResult:
        from src.data_profile import detect_column_types, generate_data_profile, load_dataset
        from src.data_split import split_dataset
        from src.evaluator import (
            build_preprocessor_from_strategy,
            evaluate_on_test,
            evaluate_strategy,
            validate_strategy,
        )
        from src.memory import (
            build_profile_summary,
            compute_profile_hash,
            compute_strategy_signature,
            create_analysis_run,
            init_db,
            query_similar_strategies,
            safe_record_attempt,
            save_analysis_report,
            save_dataset,
            save_dataset_profile,
            update_analysis_run,
            upsert_strategy,
        )
        from src.task_detection import detect_task_type

        t_start = time.time()
        cfg     = self.config
        conn    = init_db(db_path)
        run_id: Optional[int] = None

        try:
            # ----------------------------------------------------------------
            # 1. Veri yükle ve metadata'yı kaydet
            # ----------------------------------------------------------------
            df        = load_dataset(csv_path)
            task_type = detect_task_type(df, target)

            file_hash  = _file_hash(csv_path)
            dataset_id = save_dataset(conn, {
                "name":         csv_path.split("/")[-1],
                "file_path":    csv_path,
                "file_hash":    file_hash,
                "row_count":    len(df),
                "column_count": len(df.columns),
            })

            run_id = create_analysis_run(
                conn,
                dataset_id      = dataset_id,
                user_request    = user_request,
                experiment_mode = cfg.experiment_mode,
                random_seed     = cfg.random_seed,
                max_attempts    = cfg.max_attempts,
                user_id         = user_id,
            )

            # ----------------------------------------------------------------
            # 2. Train/test bölünmesi (veri sızıntısını önler)
            # ----------------------------------------------------------------
            X_train, X_test, y_train, y_test = split_dataset(
                df, target,
                task_type    = task_type,
                random_state = cfg.random_seed,
            )

            # ----------------------------------------------------------------
            # 3. Profil — yalnızca eğitim seti üzerinde
            # ----------------------------------------------------------------
            df_train = X_train.copy()
            df_train[target] = y_train.values

            profile      = generate_data_profile(df_train, target)
            col_types    = detect_column_types(df_train, target)
            numeric_cols = col_types["numeric"]
            cat_cols     = col_types["categorical"]

            profile_summary = build_profile_summary(profile, task_type)
            profile_hash    = compute_profile_hash(profile_summary)

            save_dataset_profile(conn, dataset_id, profile, profile_summary, profile_hash)

            # ----------------------------------------------------------------
            # 4. Bellek destekli mod: benzer veri setlerinden stratejileri al
            # ----------------------------------------------------------------
            memory_strategies: list[dict] = []
            if cfg.experiment_mode in ("C", "D"):
                rows = query_similar_strategies(
                    conn, profile_hash, task_type,
                    exclude_dataset_id=dataset_id,
                    limit=3,
                )
                memory_strategies = [dict(r) for r in rows]

            # ----------------------------------------------------------------
            # 5. Geri bildirim döngüsü
            # ----------------------------------------------------------------
            attempts:            list[AttemptRecord] = []
            tried_signatures:    set[str]            = set()
            best_score:          Optional[float]     = None
            best_strategy:       Optional[dict]      = None
            best_strategy_id:    Optional[int]       = None
            best_primary_metric: Optional[str]       = None
            consecutive_fails:   int                 = 0

            mem_iter = iter(memory_strategies)

            for attempt_num in range(1, cfg.max_attempts + 1):
                if time.time() - t_start > cfg.time_limit_seconds:
                    logger.info("Zaman sınırına ulaşıldı (attempt %d).", attempt_num)
                    break

                # Strateji seçimi: önce bellekten, ardından LLM'den
                strategy: Optional[dict] = None
                try:
                    mem_row  = next(mem_iter)
                    strategy = {
                        "imputation_numeric":     mem_row["imputation_numeric"],
                        "imputation_categorical": mem_row["imputation_categorical"],
                        "scaling_method":         mem_row["scaling_method"],
                        "encoding_method":        mem_row["encoding_method"],
                        "outlier_handling":       mem_row.get("outlier_handling", "none"),
                        "feature_selection":      mem_row.get("feature_selection", "none"),
                    }
                except StopIteration:
                    strategy = self.llm.suggest_strategy(
                        profile_summary, task_type, tried_signatures, attempt_num
                    )

                if strategy is None:
                    logger.info("LLM strateji önerisi tükendi (attempt %d).", attempt_num)
                    break

                sig = compute_strategy_signature(strategy)
                if sig in tried_signatures:
                    continue
                tried_signatures.add(sig)

                # Strateji doğrulama
                if not validate_strategy(strategy):
                    record = AttemptRecord(
                        attempt_number     = attempt_num,
                        strategy           = strategy,
                        strategy_id        = 0,
                        cv_primary_score   = None,
                        cv_secondary_score = None,
                        primary_metric     = "macro_f1" if task_type == "classification" else "rmse",
                        duration_seconds   = 0.0,
                        status             = "invalid",
                        error_message      = "Geçersiz strateji parametreleri.",
                    )
                    attempts.append(record)
                    consecutive_fails += 1
                    if consecutive_fails >= cfg.max_consecutive_failures:
                        break
                    continue

                strategy_id = upsert_strategy(conn, strategy)

                # CV değerlendirmesi
                t_attempt = time.time()
                try:
                    preprocessor = build_preprocessor_from_strategy(
                        strategy, numeric_cols, cat_cols
                    )
                    cv_result = evaluate_strategy(
                        X_train, y_train, preprocessor, task_type,
                        n_splits    = cfg.n_cv_splits,
                        random_seed = cfg.random_seed,
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

                    if best_score is None or _is_better(cv_score, best_score, task_type):
                        best_score          = cv_score
                        best_strategy       = strategy
                        best_strategy_id    = strategy_id
                        best_primary_metric = pmetric

                    safe_record_attempt(
                        conn, run_id, strategy_id, profile_hash, task_type,
                        {
                            "attempt_number":     attempt_num,
                            "model_type":         cv_result.get("model_type", "logistic_regression"),
                            "primary_metric_name": pmetric,
                            "cv_primary_score":   cv_score,
                            "cv_secondary_score": cv_result.get("cv_secondary_score"),
                            "cv_scores_detail":   cv_result.get("cv_scores_detail"),
                            "duration_seconds":   duration,
                            "llm_call_count":     0,
                            "status":             "success",
                        },
                        dataset_id=dataset_id,
                    )

                except Exception as exc:
                    duration = time.time() - t_attempt
                    logger.warning("Değerlendirme hatası (attempt %d): %s", attempt_num, exc)
                    pmetric = "macro_f1" if task_type == "classification" else "rmse"
                    mtype   = "logistic_regression" if task_type == "classification" else "ridge_regression"
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
                        conn, run_id, strategy_id, profile_hash, task_type,
                        {
                            "attempt_number":     attempt_num,
                            "model_type":         mtype,
                            "primary_metric_name": pmetric,
                            "cv_primary_score":   None,
                            "duration_seconds":   duration,
                            "llm_call_count":     0,
                            "status":             "failed",
                            "error_message":      str(exc),
                        },
                        dataset_id=dataset_id,
                    )
                    if consecutive_fails >= cfg.max_consecutive_failures:
                        break

                attempts.append(record)
                update_analysis_run(conn, run_id, "running", attempt_count=attempt_num)

            # ----------------------------------------------------------------
            # 6. Nihai test değerlendirmesi
            # ----------------------------------------------------------------
            total_duration = time.time() - t_start
            final_scores: Optional[dict] = None

            if best_strategy is not None:
                try:
                    preprocessor = build_preprocessor_from_strategy(
                        best_strategy, numeric_cols, cat_cols
                    )
                    final_scores = evaluate_on_test(
                        X_train, y_train, X_test, y_test,
                        preprocessor, task_type,
                        random_seed=cfg.random_seed,
                    )
                except Exception as exc:
                    logger.error("Test değerlendirme hatası: %s", exc)

            # ----------------------------------------------------------------
            # 7. Rapor ve durum kaydı
            # ----------------------------------------------------------------
            if best_strategy is not None:
                update_analysis_run(conn, run_id, "completed", attempt_count=len(attempts))
                save_analysis_report(
                    conn,
                    run_id               = run_id,
                    best_strategy_id     = best_strategy_id,
                    final_scores         = final_scores or {},
                    cv_primary_score     = best_score,
                    total_attempts       = len(attempts),
                    total_duration_seconds = total_duration,
                    total_llm_calls      = 0,
                )
                result_status = "completed"
            else:
                update_analysis_run(
                    conn, run_id, "failed",
                    attempt_count=len(attempts),
                    error_message="Geçerli strateji bulunamadı.",
                )
                result_status = "failed"

            conn.close()
            return AnalysisResult(
                status              = result_status,
                task_type           = task_type,
                best_strategy       = best_strategy,
                best_cv_score       = best_score,
                primary_metric      = best_primary_metric,
                final_test_scores   = final_scores,
                attempts            = attempts,
                total_duration_secs = total_duration,
                run_id              = run_id,
            )

        except Exception as exc:
            logger.error("Orkestrasyon hatası: %s", exc, exc_info=True)
            if run_id is not None:
                try:
                    update_analysis_run(conn, run_id, "failed", error_message=str(exc))
                except Exception:
                    pass
            try:
                conn.close()
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
                total_duration_secs = time.time() - t_start,
                run_id              = run_id,
                error_message       = str(exc),
            )


# ---------------------------------------------------------------------------
# Özel yardımcılar
# ---------------------------------------------------------------------------

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_better(new_score: float, best_score: float, task_type: str) -> bool:
    if task_type == "classification":
        return new_score > best_score
    return new_score < best_score  # regression: daha düşük RMSE daha iyi
