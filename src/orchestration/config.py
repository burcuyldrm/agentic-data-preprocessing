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
    max_llm_calls:            int   = 15   # AI-03: gerçek sağlayıcı çağrı bütçesi

    def __post_init__(self) -> None:
        if not isinstance(self.max_llm_calls, int) or not (1 <= self.max_llm_calls <= 15):
            raise ValueError(
                f"max_llm_calls 1 ile 15 arasında tam sayı olmalı, "
                f"{self.max_llm_calls!r} alındı."
            )
