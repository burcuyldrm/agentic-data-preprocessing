"""
FR-07 — Analiz çalıştırma endpoint'i.

POST /analysis/run — CSV dosyasını alır, önişleme orkestrasyonunu başlatır
                     ve en iyi stratejiyi döndürür.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

import src.config as config
from src.api.dependencies import get_current_user
from src.orchestrator import Orchestrator, OrchestratorConfig

router = APIRouter(prefix="/analysis", tags=["analysis"])

_VALID_MODES = {"A", "B", "C", "D"}


class RunResponse(BaseModel):
    status:              str
    run_id:              Optional[int]
    task_type:           str
    best_strategy:       Optional[dict]
    best_cv_score:       Optional[float]
    primary_metric:      Optional[str]
    final_test_scores:   Optional[dict]
    total_attempts:      int
    total_duration_secs: float
    total_llm_calls:     int  = 0
    llm_fallback_used:   bool = False
    error_message:       Optional[str] = None


@router.post("/run", response_model=RunResponse, status_code=status.HTTP_200_OK)
def run_analysis(
    file:            UploadFile = File(...),
    target:          str        = Form(...),
    experiment_mode: str        = Form("C"),
    max_attempts:    int        = Form(5),
    random_seed:     int        = Form(42),
    current_user: dict = Depends(get_current_user),
):
    """
    CSV dosyasını alır, önişleme orkestrasyonunu çalıştırır
    ve en iyi stratejiyi döndürür.

    - `file`            : CSV dosyası (multipart/form-data)
    - `target`          : Hedef sütun adı
    - `experiment_mode` : A/B/C/D (varsayılan C)
    - `max_attempts`    : Maksimum deneme sayısı (1–10)
    - `random_seed`     : Rastgelelik tohumu
    """
    if experiment_mode not in _VALID_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Geçersiz experiment_mode. Geçerli değerler: {sorted(_VALID_MODES)}",
        )
    if not (1 <= max_attempts <= 10):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="max_attempts 1 ile 10 arasında olmalıdır.",
        )

    suffix = os.path.splitext(file.filename or "upload.csv")[1] or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        cfg    = OrchestratorConfig(
            max_attempts    = max_attempts,
            experiment_mode = experiment_mode,
            random_seed     = random_seed,
        )
        orch   = Orchestrator(config=cfg)
        result = orch.run(
            csv_path     = tmp_path,
            target       = target,
            db_path      = config.DB_PATH,
            user_id      = current_user["id"],
        )
    finally:
        os.unlink(tmp_path)

    return RunResponse(
        status              = result.status,
        run_id              = result.run_id,
        task_type           = result.task_type,
        best_strategy       = result.best_strategy,
        best_cv_score       = result.best_cv_score,
        primary_metric      = result.primary_metric,
        final_test_scores   = result.final_test_scores,
        total_attempts      = len(result.attempts),
        total_duration_secs = result.total_duration_secs,
        total_llm_calls     = result.total_llm_calls,
        llm_fallback_used   = result.llm_fallback_used,
        error_message       = result.error_message,
    )
