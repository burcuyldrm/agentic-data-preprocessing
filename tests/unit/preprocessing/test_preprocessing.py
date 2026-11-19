"""
Karakterizasyon testleri — src.strategy_selector ve src.preprocessing.

Bu testler mevcut davranışı belgeler; yeni davranış tanımlamaz.
Testler kırılırsa değiştirilen kodun gözlemlenen davranışı bozduğu anlaşılır.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import AdaptivePreprocessor, build_baseline_preprocessor
from src.strategy_selector import select_preprocessing_strategy


# ---------------------------------------------------------------------------
# select_preprocessing_strategy
# ---------------------------------------------------------------------------


class TestSelectPreprocessingStrategy:
    def _df(self, values, col="a"):
        return pd.DataFrame({col: values})

    def test_high_skewness_uses_robust_and_median(self):
        # Yüksek çarpıklık (abs >= 1): robust scaler + median imputer
        df = self._df([1.0, 1.0, 1.0, 1.0, 100.0])
        result = select_preprocessing_strategy(df, ["a"])
        assert result["a"]["imputer"] == "median"
        assert result["a"]["scaler"] == "robust"

    def test_negative_high_skewness_uses_robust(self):
        # Negatif yüksek çarpıklık da abs >= 1 koşulunu sağlar
        df = self._df([100.0, 100.0, 100.0, 100.0, 1.0])
        result = select_preprocessing_strategy(df, ["a"])
        assert result["a"]["scaler"] == "robust"

    def test_low_skewness_uses_standard_and_mean(self):
        # Düşük çarpıklık (abs < 1): standard scaler + mean imputer
        df = self._df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_preprocessing_strategy(df, ["a"])
        assert result["a"]["imputer"] == "mean"
        assert result["a"]["scaler"] == "standard"

    def test_zero_skewness_uses_standard_and_mean(self):
        # Tam simetrik dağılım: skewness = 0 → standard + mean
        df = self._df([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = select_preprocessing_strategy(df, ["a"])
        assert result["a"]["imputer"] == "mean"
        assert result["a"]["scaler"] == "standard"
        assert result["a"]["skewness"] == 0.0

    def test_nan_skewness_uses_median_and_standard(self):
        # Tüm değerler NaN → skewness NaN → median + standard
        df = self._df([float("nan")] * 5)
        result = select_preprocessing_strategy(df, ["a"])
        assert result["a"]["imputer"] == "median"
        assert result["a"]["scaler"] == "standard"
        assert result["a"]["skewness"] is None

    def test_skewness_is_none_for_nan_column(self):
        df = self._df([float("nan")] * 3)
        result = select_preprocessing_strategy(df, ["a"])
        assert result["a"]["skewness"] is None

    def test_skewness_rounded_to_4_decimal_places(self):
        # Çarpıklık değeri 4 ondalık basamağa yuvarlanmalı
        df = self._df([1.0, 2.0, 3.0, 4.0, 10.0])
        result = select_preprocessing_strategy(df, ["a"])
        skew = result["a"]["skewness"]
        if skew is not None:
            assert skew == round(skew, 4)

    def test_empty_column_list_returns_empty_dict(self):
        df = self._df([1.0, 2.0, 3.0])
        result = select_preprocessing_strategy(df, [])
        assert result == {}

    def test_multiple_columns_evaluated_independently(self):
        # Her sütun bağımsız olarak değerlendirilmeli
        df = pd.DataFrame({
            "a": [1.0, 1.0, 1.0, 1.0, 100.0],  # yüksek çarpıklık
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],      # düşük çarpıklık
        })
        result = select_preprocessing_strategy(df, ["a", "b"])
        assert result["a"]["scaler"] == "robust"
        assert result["b"]["scaler"] == "standard"
        assert result["a"]["imputer"] == "median"
        assert result["b"]["imputer"] == "mean"

    def test_result_contains_required_keys(self):
        df = self._df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = select_preprocessing_strategy(df, ["a"])
        assert set(result["a"].keys()) == {"skewness", "imputer", "scaler"}

    def test_all_columns_appear_in_result(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0]})
        result = select_preprocessing_strategy(df, ["x", "y", "z"])
        assert set(result.keys()) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# build_baseline_preprocessor
# ---------------------------------------------------------------------------


class TestBuildBaselinePreprocessor:
    def test_returns_column_transformer(self):
        result = build_baseline_preprocessor(["a"], ["c"])
        assert isinstance(result, ColumnTransformer)

    def test_transformer_names(self):
        result = build_baseline_preprocessor(["a", "b"], ["c"])
        names = [name for name, _, _ in result.transformers]
        assert "numeric" in names
        assert "categorical" in names

    def test_numeric_pipeline_uses_median_imputer(self):
        result = build_baseline_preprocessor(["a"], [])
        numeric_pipe = dict((name, t) for name, t, _ in result.transformers)["numeric"]
        assert isinstance(numeric_pipe, Pipeline)
        imputer = numeric_pipe.named_steps["imputer"]
        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "median"

    def test_numeric_pipeline_uses_standard_scaler(self):
        result = build_baseline_preprocessor(["a"], [])
        numeric_pipe = dict((name, t) for name, t, _ in result.transformers)["numeric"]
        assert isinstance(numeric_pipe.named_steps["scaler"], StandardScaler)

    def test_categorical_pipeline_uses_most_frequent_imputer(self):
        result = build_baseline_preprocessor([], ["c"])
        cat_pipe = dict((name, t) for name, t, _ in result.transformers)["categorical"]
        assert isinstance(cat_pipe, Pipeline)
        assert cat_pipe.named_steps["imputer"].strategy == "most_frequent"

    def test_categorical_pipeline_uses_onehot_encoder(self):
        result = build_baseline_preprocessor([], ["c"])
        cat_pipe = dict((name, t) for name, t, _ in result.transformers)["categorical"]
        ohe = cat_pipe.named_steps["encoder"]
        assert isinstance(ohe, OneHotEncoder)
        assert ohe.handle_unknown == "ignore"
        assert ohe.sparse_output is False

    def test_numeric_columns_assigned_to_numeric_transformer(self):
        cols = ["a", "b"]
        result = build_baseline_preprocessor(cols, ["c"])
        col_map = {name: c for name, _, c in result.transformers}
        assert col_map["numeric"] == cols

    def test_categorical_columns_assigned_to_categorical_transformer(self):
        cat_cols = ["c", "d"]
        result = build_baseline_preprocessor(["a"], cat_cols)
        col_map = {name: c for name, _, c in result.transformers}
        assert col_map["categorical"] == cat_cols

    def test_fit_transform_produces_correct_row_count(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
            "c": ["x", "y", "x"],
        })
        prep = build_baseline_preprocessor(["a", "b"], ["c"])
        result = prep.fit_transform(df)
        assert result.shape[0] == 3

    def test_fit_transform_output_is_array(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", "x"]})
        prep = build_baseline_preprocessor(["a"], ["c"])
        result = prep.fit_transform(df)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# AdaptivePreprocessor
# ---------------------------------------------------------------------------


class TestAdaptivePreprocessor:
    def _make_df(self):
        return pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],      # düşük çarpıklık
            "b": [1.0, 1.0, 1.0, 1.0, 100.0],      # yüksek çarpıklık
            "c": ["x", "y", "x", "y", "x"],
        })

    def test_is_base_estimator(self):
        prep = AdaptivePreprocessor(["a"], ["c"])
        assert isinstance(prep, BaseEstimator)

    def test_is_transformer_mixin(self):
        prep = AdaptivePreprocessor(["a"], ["c"])
        assert isinstance(prep, TransformerMixin)

    def test_fit_sets_strategies_attribute(self):
        df = self._make_df()
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        prep.fit(df[["a", "b", "c"]])
        assert hasattr(prep, "strategies_")
        assert "a" in prep.strategies_
        assert "b" in prep.strategies_

    def test_high_skew_column_selects_robust(self):
        df = self._make_df()
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        prep.fit(df[["a", "b", "c"]])
        assert prep.strategies_["b"]["scaler"] == "robust"

    def test_low_skew_column_selects_standard(self):
        df = self._make_df()
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        prep.fit(df[["a", "b", "c"]])
        assert prep.strategies_["a"]["scaler"] == "standard"

    def test_fit_sets_preprocessor_attribute(self):
        df = self._make_df()
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        prep.fit(df[["a", "b", "c"]])
        assert hasattr(prep, "preprocessor_")
        assert isinstance(prep.preprocessor_, ColumnTransformer)

    def test_fit_transform_produces_correct_row_count(self):
        df = self._make_df()
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        result = prep.fit_transform(df[["a", "b", "c"]])
        assert result.shape[0] == 5

    def test_fit_transform_output_is_array(self):
        df = self._make_df()
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        result = prep.fit_transform(df[["a", "b", "c"]])
        assert isinstance(result, np.ndarray)

    def test_transform_on_unseen_data_preserves_shape(self):
        train = self._make_df()
        test = pd.DataFrame({
            "a": [6.0, 7.0],
            "b": [2.0, 3.0],
            "c": ["x", "y"],
        })
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        prep.fit(train[["a", "b", "c"]])
        result = prep.transform(test[["a", "b", "c"]])
        assert result.shape[0] == 2

    def test_transform_column_count_consistent(self):
        # Train ve test aynı sütun sayısı üretmeli
        train = self._make_df()
        test = pd.DataFrame({
            "a": [6.0, 7.0],
            "b": [2.0, 3.0],
            "c": ["x", "y"],
        })
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        train_result = prep.fit_transform(train[["a", "b", "c"]])
        test_result  = prep.transform(test[["a", "b", "c"]])
        assert train_result.shape[1] == test_result.shape[1]

    def test_numeric_columns_stored_on_instance(self):
        prep = AdaptivePreprocessor(["a", "b"], ["c"])
        assert prep.numeric_columns == ["a", "b"]

    def test_categorical_columns_stored_on_instance(self):
        prep = AdaptivePreprocessor(["a"], ["c", "d"])
        assert prep.categorical_columns == ["c", "d"]
