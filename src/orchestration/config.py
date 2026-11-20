"""
Orkestrasyon çalıştırma yapılandırması.
"""

from dataclasses import dataclass


@dataclass
class OrchestratorConfig:
    max_attempts:             int   = 5
    max_consecutive_failures: int   = 2
    time_limit_seconds:       float = 600.0
    random_seed:              int   = 42
    experiment_mode:          str   = "C"
    n_cv_splits:              int   = 5
