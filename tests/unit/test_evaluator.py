"""
FR-07 — Değerlendirici (evaluator) birim testleri.

Kabul kriterleri (FR-07):
  - Strateji doğrulama tüm geçerli / geçersiz kombinasyonları ayırt eder.
  - build_preprocessor_from_strategy her desteklenen seçenek için doğru
    sklearn bileşenlerini içeren bir ColumnTransformer döndürür.
  - evaluate_strategy sınıflandırma için macro_f1, regresyon için rmse
    döndürür; cv_scores_detail ve duration_seconds her zaman vardır.
  - evaluate_on_test nihai test metriklerini döndürür; test setini sadece
    bir kez kullanır (strateji seçimine dahil etmez).
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluator import (
    VALID_ENCODING_METHOD,
    VALID_IMPUTATION_NUMERIC,
    VALID_SCALING_METHOD,
    build_preprocessor_from_strategy,
    evaluate_on_test,
    evaluate_strategy,
    validate_strategy,
)

_VALID = {
    "imputation_numeric":     "median",
    "imputation_categorical": "most_frequent",
    "scaling_method":         "standard",
    "encoding_method":        "onehot",
    "outlier_handling":       "none",
    "feature_selection":      "none",
}


# ---------------------------------------------------------------------------
# Veri yardımcıları
# ---------------------------------------------------------------------------

def _clf_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.normal(size=n),
        "cat1": rng.choice(["a", "b", "c"], size=n),
    })
    y = pd.Series(rng.integers(0, 2, size=n).astype(str), name="target")
    return X, y


def _reg_data(n: int = 200, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "num1": rng.normal(size=n),
        "num2": rng.normal(size=n),
    })
    y = pd.Series(rng.normal(size=n), name="target")
    return X, y


# ---------------------------------------------------------------------------
# TestValidateStrategy
# ---------------------------------------------------------------------------

class TestValidateStrategy:
    def test_valid_strategy_returns_true(self):
        assert validate_strategy(_VALID) is True

    def test_missing_imputation_numeric_returns_false(self):
        s = {k: v for k, v in _VALID.items() if k != "imputation_numeric"}
        assert validate_strategy(s) is False

    def test_missing_imputation_categorical_returns_false(self):
        s = {k: v for k, v in _VALID.items() if k != "imputation_categorical"}
        assert validate_strategy(s) is False

    def test_missing_scaling_method_returns_false(self):
        s = {k: v for k, v in _VALID.items() if k != "scaling_method"}
        assert validate_strategy(s) is False

    def test_missing_encoding_method_returns_false(self):
        s = {k: v for k, v in _VALID.items() if k != "encoding_method"}
        assert validate_strategy(s) is False

    def test_invalid_imputation_numeric(self):
        assert validate_strategy({**_VALID, "imputation_numeric": "iterative"}) is False

    def test_invalid_imputation_categorical(self):
        assert validate_strategy({**_VALID, "imputation_categorical": "knn"}) is False

    def test_invalid_scaling_method(self):
        assert validate_strategy({**_VALID, "scaling_method": "log"}) is False

    def test_invalid_encoding_method(self):
        assert validate_strategy({**_VALID, "encoding_method": "binary"}) is False

    def test_invalid_outlier_handling(self):
        assert validate_strategy({**_VALID, "outlier_handling": "winsorize"}) is False

    def test_invalid_feature_selection(self):
        assert validate_strategy({**_VALID, "feature_selection": "pca"}) is False

    def test_all_valid_imputation_numeric(self):
        for m in VALID_IMPUTATION_NUMERIC:
            assert validate_strategy({**_VALID, "imputation_numeric": m}) is True

    def test_all_valid_scaling_methods(self):
        for m in VALID_SCALING_METHOD:
            assert validate_strategy({**_VALID, "scaling_method": m}) is True

    def test_all_valid_outlier_handling(self):
        for m in ("clip", "remove", "none"):
            assert validate_strategy({**_VALID, "outlier_handling": m}) is True

    def test_all_valid_feature_selection(self):
        for m in ("none", "variance_threshold", "selectkbest"):
            assert validate_strategy({**_VALID, "feature_selection": m}) is True

    def test_empty_dict_returns_false(self):
        assert validate_strategy({}) is False


# ---------------------------------------------------------------------------
# TestBuildPreprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_returns_column_transformer(self):
        from sklearn.compose import ColumnTransformer

        prep = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        assert isinstance(prep, ColumnTransformer)

    def test_fit_transform_shape(self):
        X, _ = _clf_data()
        prep = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = prep.fit_transform(X)
        assert result.shape[0] == len(X)

    def test_knn_imputer_created(self):
        from sklearn.impute import KNNImputer

        s    = {**_VALID, "imputation_numeric": "knn"}
        prep = build_preprocessor_from_strategy(s, ["num1"], [])
        num_pipeline = prep.transformers[0][1]
        assert isinstance(num_pipeline.steps[0][1], KNNImputer)

    def test_knn_neighbors_from_extra_params(self):
        from sklearn.impute import KNNImputer

        s    = {**_VALID, "imputation_numeric": "knn", "extra_params": {"knn_neighbors": 7}}
        prep = build_preprocessor_from_strategy(s, ["num1"], [])
        imputer = prep.transformers[0][1].steps[0][1]
        assert imputer.n_neighbors == 7

    def test_robust_scaler_created(self):
        from sklearn.preprocessing import RobustScaler

        s    = {**_VALID, "scaling_method": "robust"}
        prep = build_preprocessor_from_strategy(s, ["num1"], [])
        num_pipeline = prep.transformers[0][1]
        assert isinstance(num_pipeline.steps[1][1], RobustScaler)

    def test_minmax_scaler_created(self):
        from sklearn.preprocessing import MinMaxScaler

        s    = {**_VALID, "scaling_method": "minmax"}
        prep = build_preprocessor_from_strategy(s, ["num1"], [])
        num_pipeline = prep.transformers[0][1]
        assert isinstance(num_pipeline.steps[1][1], MinMaxScaler)

    def test_no_scaler_step_when_none(self):
        s    = {**_VALID, "scaling_method": "none"}
        prep = build_preprocessor_from_strategy(s, ["num1"], [])
        num_pipeline = prep.transformers[0][1]
        assert len(num_pipeline.steps) == 1

    def test_numeric_only_transforms(self):
        X, _ = _reg_data()
        prep = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = prep.fit_transform(X)
        assert result.shape[0] == len(X)

    def test_categorical_only_transforms(self):
        df   = pd.DataFrame({"cat1": ["a", "b", "a", "c"]})
        prep = build_preprocessor_from_strategy(_VALID, [], ["cat1"])
        result = prep.fit_transform(df)
        assert result.shape[0] == 4

    def test_missing_values_imputed(self):
        X, _       = _clf_data()
        X = X.copy()
        X.loc[0, "num1"] = float("nan")
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = prep.fit_transform(X)
        assert not np.isnan(result).any()

    def test_categorical_constant_imputer(self):
        from sklearn.impute import SimpleImputer

        s    = {**_VALID, "imputation_categorical": "constant"}
        prep = build_preprocessor_from_strategy(s, [], ["cat1"])
        cat_pipeline = prep.transformers[0][1]
        imputer = cat_pipeline.steps[0][1]
        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "constant"


# ---------------------------------------------------------------------------
# TestEvaluateStrategy
# ---------------------------------------------------------------------------

class TestEvaluateStrategy:
    def test_classification_primary_metric_name(self):
        X, y   = _clf_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_strategy(X, y, prep, "classification", n_splits=3)
        assert result["primary_metric_name"] == "macro_f1"

    def test_classification_primary_score_range(self):
        X, y   = _clf_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_strategy(X, y, prep, "classification", n_splits=3)
        assert 0.0 <= result["cv_primary_score"] <= 1.0

    def test_regression_primary_metric_name(self):
        X, y   = _reg_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_strategy(X, y, prep, "regression", n_splits=3)
        assert result["primary_metric_name"] == "rmse"

    def test_regression_primary_score_positive(self):
        X, y   = _reg_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_strategy(X, y, prep, "regression", n_splits=3)
        assert result["cv_primary_score"] > 0.0

    def test_cv_scores_detail_has_fold_scores(self):
        X, y   = _clf_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_strategy(X, y, prep, "classification", n_splits=3)
        assert "fold_scores" in result["cv_scores_detail"]
        assert len(result["cv_scores_detail"]["fold_scores"]) == 3

    def test_cv_scores_detail_has_mean_std(self):
        X, y   = _clf_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_strategy(X, y, prep, "classification", n_splits=3)
        detail = result["cv_scores_detail"]
        assert "mean" in detail
        assert "std" in detail

    def test_duration_seconds_non_negative(self):
        X, y   = _clf_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_strategy(X, y, prep, "classification", n_splits=3)
        assert result["duration_seconds"] >= 0.0

    def test_model_type_classification(self):
        X, y   = _clf_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_strategy(X, y, prep, "classification", n_splits=3)
        assert result["model_type"] == "logistic_regression"

    def test_model_type_regression(self):
        X, y   = _reg_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_strategy(X, y, prep, "regression", n_splits=3)
        assert result["model_type"] == "ridge_regression"

    def test_secondary_metric_classification_range(self):
        X, y   = _clf_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_strategy(X, y, prep, "classification", n_splits=3)
        assert result["cv_secondary_score"] is not None
        assert 0.0 <= result["cv_secondary_score"] <= 1.0

    def test_secondary_metric_regression_positive(self):
        X, y   = _reg_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_strategy(X, y, prep, "regression", n_splits=3)
        assert result["cv_secondary_score"] is not None
        assert result["cv_secondary_score"] > 0.0

    def test_regression_detail_includes_fold_mae(self):
        X, y   = _reg_data()
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_strategy(X, y, prep, "regression", n_splits=3)
        assert "fold_mae" in result["cv_scores_detail"]


# ---------------------------------------------------------------------------
# TestEvaluateOnTest
# ---------------------------------------------------------------------------

class TestEvaluateOnTest:
    def _split(self, X, y, ratio=0.8):
        n = int(ratio * len(X))
        return X.iloc[:n], X.iloc[n:], y.iloc[:n], y.iloc[n:]

    def test_classification_primary_score_range(self):
        X, y = _clf_data(n=200)
        X_tr, X_te, y_tr, y_te = self._split(X, y)
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_on_test(X_tr, y_tr, X_te, y_te, prep, "classification")
        assert 0.0 <= result["final_test_primary_score"] <= 1.0

    def test_classification_secondary_score_range(self):
        X, y = _clf_data(n=200)
        X_tr, X_te, y_tr, y_te = self._split(X, y)
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_on_test(X_tr, y_tr, X_te, y_te, prep, "classification")
        assert 0.0 <= result["final_test_secondary_score"] <= 1.0

    def test_regression_primary_positive(self):
        X, y = _reg_data(n=200)
        X_tr, X_te, y_tr, y_te = self._split(X, y)
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_on_test(X_tr, y_tr, X_te, y_te, prep, "regression")
        assert result["final_test_primary_score"] > 0.0

    def test_metrics_detail_included_classification(self):
        X, y = _clf_data(n=200)
        X_tr, X_te, y_tr, y_te = self._split(X, y)
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_on_test(X_tr, y_tr, X_te, y_te, prep, "classification")
        assert "final_test_metrics_detail" in result
        assert "macro_f1" in result["final_test_metrics_detail"]
        assert "balanced_accuracy" in result["final_test_metrics_detail"]

    def test_metrics_detail_included_regression(self):
        X, y = _reg_data(n=200)
        X_tr, X_te, y_tr, y_te = self._split(X, y)
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_on_test(X_tr, y_tr, X_te, y_te, prep, "regression")
        assert "final_test_metrics_detail" in result
        assert "rmse" in result["final_test_metrics_detail"]
        assert "mae" in result["final_test_metrics_detail"]

    def test_primary_metric_name_classification(self):
        X, y = _clf_data(n=200)
        X_tr, X_te, y_tr, y_te = self._split(X, y)
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], ["cat1"])
        result = evaluate_on_test(X_tr, y_tr, X_te, y_te, prep, "classification")
        assert result["primary_metric_name"] == "macro_f1"

    def test_primary_metric_name_regression(self):
        X, y = _reg_data(n=200)
        X_tr, X_te, y_tr, y_te = self._split(X, y)
        prep   = build_preprocessor_from_strategy(_VALID, ["num1", "num2"], [])
        result = evaluate_on_test(X_tr, y_tr, X_te, y_te, prep, "regression")
        assert result["primary_metric_name"] == "rmse"
