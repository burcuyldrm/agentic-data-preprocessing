"""
Mod A/B yardımcı fonksiyon ve sabit birim testleri.

_BASELINE_STRATEGY geçerliliği ve _derive_strategy_from_profile mantığı:
çoğunluk oyu, eşitlik kuralları, boş sütun ve NaN skewness kenar durumları.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import validate_strategy
from src.orchestration.config import OrchestratorConfig
from src.orchestration.runner import Orchestrator, _BASELINE_STRATEGY, _RunContext


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _make_ctx(X_train: pd.DataFrame, numeric_cols: list) -> _RunContext:
    ctx = _RunContext(conn=None, cfg=OrchestratorConfig(), t_start=0.0)  # type: ignore[arg-type]
    ctx.X_train = X_train
    ctx.numeric_cols = numeric_cols
    ctx.cat_cols = []
    return ctx


def _skewed_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Tüm sütunlar yüksek çarpıklık (|skewness| >> 1)."""
    rng = np.random.default_rng(seed)
    base = rng.exponential(scale=1.0, size=n)
    return pd.DataFrame({
        "feat_a": base ** 3,
        "feat_b": base ** 4,
    })


def _normal_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Tüm sütunlar düşük çarpıklık (normal dağılım)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
    })


# ---------------------------------------------------------------------------
# _BASELINE_STRATEGY
# ---------------------------------------------------------------------------

class TestBaselineStrategy:
    def test_baseline_strategy_is_valid(self):
        assert validate_strategy(_BASELINE_STRATEGY) is True

    def test_baseline_strategy_has_required_keys(self):
        required = {
            "imputation_numeric", "imputation_categorical",
            "scaling_method", "encoding_method",
            "outlier_handling", "feature_selection",
        }
        assert required.issubset(_BASELINE_STRATEGY.keys())

    def test_baseline_scaling_is_standard(self):
        assert _BASELINE_STRATEGY["scaling_method"] == "standard"

    def test_baseline_imputation_is_median(self):
        assert _BASELINE_STRATEGY["imputation_numeric"] == "median"


# ---------------------------------------------------------------------------
# _derive_strategy_from_profile
# ---------------------------------------------------------------------------

class TestDeriveStrategyFromProfile:
    def _derive(self, df: pd.DataFrame, cols: list) -> dict:
        ctx = _make_ctx(df, cols)
        return Orchestrator()._derive_strategy_from_profile(ctx)

    def test_high_skew_all_columns_gives_robust(self):
        df = _skewed_df()
        result = self._derive(df, ["feat_a", "feat_b"])
        assert result["scaling_method"] == "robust"

    def test_high_skew_all_columns_gives_median(self):
        df = _skewed_df()
        result = self._derive(df, ["feat_a", "feat_b"])
        assert result["imputation_numeric"] == "median"

    def test_low_skew_all_columns_gives_standard(self):
        df = _normal_df()
        result = self._derive(df, ["feat_a", "feat_b"])
        assert result["scaling_method"] == "standard"

    def test_low_skew_all_columns_gives_mean(self):
        df = _normal_df()
        result = self._derive(df, ["feat_a", "feat_b"])
        assert result["imputation_numeric"] == "mean"

    def test_empty_numeric_cols_returns_baseline(self):
        df = pd.DataFrame({"cat": ["a", "b", "c"]})
        result = self._derive(df, [])
        assert result == _BASELINE_STRATEGY

    def test_majority_robust_wins(self):
        rng = np.random.default_rng(0)
        base = rng.exponential(scale=1.0, size=60) ** 3  # skewed
        normal = rng.normal(size=60)
        df = pd.DataFrame({
            "skew1": base,
            "skew2": base ** 2,
            "skew3": base,
            "norm1": normal,
        })
        result = self._derive(df, ["skew1", "skew2", "skew3", "norm1"])
        assert result["scaling_method"] == "robust"

    def test_majority_standard_wins(self):
        rng = np.random.default_rng(0)
        base = rng.exponential(scale=1.0, size=60) ** 3
        normal = rng.normal(size=60)
        df = pd.DataFrame({
            "skew1": base,
            "norm1": normal,
            "norm2": normal,
            "norm3": normal,
        })
        result = self._derive(df, ["skew1", "norm1", "norm2", "norm3"])
        assert result["scaling_method"] == "standard"

    def test_tie_scaling_gives_standard(self):
        """Eşit dağılımda "standard" seçilir (muhafazakâr tercih)."""
        rng = np.random.default_rng(0)
        base = rng.exponential(scale=1.0, size=60) ** 3
        normal = rng.normal(size=60)
        df = pd.DataFrame({
            "skew1": base,
            "norm1": normal,
        })
        result = self._derive(df, ["skew1", "norm1"])
        assert result["scaling_method"] == "standard"

    def test_tie_imputation_gives_median(self):
        """Eşit dağılımda imputation "median" seçilir (eşitlikte median >= mean)."""
        rng = np.random.default_rng(0)
        base = rng.exponential(scale=1.0, size=60) ** 3
        normal = rng.normal(size=60)
        df = pd.DataFrame({
            "skew1": base,
            "norm1": normal,
        })
        result = self._derive(df, ["skew1", "norm1"])
        assert result["imputation_numeric"] == "median"

    def test_nan_skewness_column_gives_median_standard(self):
        """Tüm değerler aynıysa skewness NaN → imputer=median, scaler=standard."""
        df = pd.DataFrame({"const": [5.0] * 20, "normal": np.random.default_rng(0).normal(size=20)})
        result = self._derive(df, ["const", "normal"])
        assert result["imputation_numeric"] in ("median",)

    def test_returned_strategy_is_valid(self):
        df = _skewed_df()
        result = self._derive(df, ["feat_a", "feat_b"])
        assert validate_strategy(result) is True

    def test_returned_strategy_has_fixed_categorical(self):
        df = _skewed_df()
        result = self._derive(df, ["feat_a", "feat_b"])
        assert result["imputation_categorical"] == "most_frequent"
        assert result["encoding_method"] == "onehot"
        assert result["outlier_handling"] == "none"
        assert result["feature_selection"] == "none"

    def test_strategy_copy_does_not_mutate_baseline(self):
        """_BASELINE_STRATEGY sabiti değişmemeli."""
        original = dict(_BASELINE_STRATEGY)
        df = _normal_df()
        self._derive(df, ["feat_a", "feat_b"])
        assert _BASELINE_STRATEGY == original
