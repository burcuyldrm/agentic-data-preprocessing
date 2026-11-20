"""
Orkestrasyon veri transfer nesneleri.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
