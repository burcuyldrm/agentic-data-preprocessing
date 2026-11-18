"""
FR-07 — Strateji değerlendirici.

Dışa açık fonksiyonlar:
  validate_strategy                — Strateji parametrelerini doğrular.
  build_preprocessor_from_strategy — ColumnTransformer oluşturur.
  evaluate_strategy                — 5 katlı CV ile skor hesaplar.
  evaluate_on_test                 — Nihai test değerlendirmesi (bir kez çağrılır).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

VALID_IMPUTATION_NUMERIC     = {"mean", "median", "knn"}
VALID_IMPUTATION_CATEGORICAL = {"most_frequent", "constant"}
VALID_SCALING_METHOD         = {"standard", "robust", "minmax", "none"}
VALID_ENCODING_METHOD        = {"onehot"}
VALID_OUTLIER_HANDLING       = {"clip", "remove", "none"}
VALID_FEATURE_SELECTION      = {"none", "variance_threshold", "selectkbest"}


def validate_strategy(strategy: dict) -> bool:
    return (
        strategy.get("imputation_numeric")     in VALID_IMPUTATION_NUMERIC
        and strategy.get("imputation_categorical") in VALID_IMPUTATION_CATEGORICAL
        and strategy.get("scaling_method")     in VALID_SCALING_METHOD
        and strategy.get("encoding_method")    in VALID_ENCODING_METHOD
        and strategy.get("outlier_handling",  "none") in VALID_OUTLIER_HANDLING
        and strategy.get("feature_selection", "none") in VALID_FEATURE_SELECTION
    )


def build_preprocessor_from_strategy(
    strategy: dict,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    imp_num = strategy["imputation_numeric"]
    imp_cat = strategy["imputation_categorical"]
    scaling = strategy["scaling_method"]
    extra   = strategy.get("extra_params") or {}

    # Sayısal imputer
    if imp_num == "knn":
        knn_n       = int(extra.get("knn_neighbors", 5))
        num_imputer = KNNImputer(n_neighbors=knn_n)
    else:
        num_imputer = SimpleImputer(strategy=imp_num)

    # Ölçekleyici
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None  # none

    num_steps: list = [("imputer", num_imputer)]
    if scaler is not None:
        num_steps.append(("scaler", scaler))

    # Kategorik imputer
    if imp_cat == "constant":
        cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    else:
        cat_imputer = SimpleImputer(strategy="most_frequent")

    cat_steps = [
        ("imputer", cat_imputer),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def evaluate_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    task_type: str,
    n_splits: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    5 katlı çapraz doğrulama ile strateji performansını ölçer.

    Her katlama için pipeline klonlanır — preprocessor nesnesi
    katlamalar arasında paylaşılmaz; durum kirlenmesi olmaz.
    """
    t0 = time.time()

    if task_type == "classification":
        model    = LogisticRegression(max_iter=1000, random_state=random_seed)
        cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        fold_scores: list[float] = []
        all_y_true: list = []
        all_y_pred: list = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr  = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr  = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            p = clone(pipeline)
            p.fit(X_tr, y_tr)
            yp = p.predict(X_val)

            fold_scores.append(
                float(f1_score(y_val, yp, average="macro", zero_division=0))
            )
            all_y_true.extend(y_val.tolist())
            all_y_pred.extend(yp.tolist())

        cv_primary   = float(np.mean(fold_scores))
        cv_secondary = float(balanced_accuracy_score(all_y_true, all_y_pred))

        return {
            "primary_metric_name":  "macro_f1",
            "cv_primary_score":     cv_primary,
            "cv_secondary_score":   cv_secondary,
            "cv_scores_detail": {
                "fold_scores": fold_scores,
                "mean":        cv_primary,
                "std":         float(np.std(fold_scores)),
            },
            "model_type":       "logistic_regression",
            "duration_seconds": time.time() - t0,
        }

    else:  # regression
        model    = Ridge()
        cv       = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        fold_rmse: list[float] = []
        fold_mae:  list[float] = []

        for train_idx, val_idx in cv.split(X_train):
            X_tr  = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr  = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            p = clone(pipeline)
            p.fit(X_tr, y_tr)
            yp = p.predict(X_val)

            fold_rmse.append(float(np.sqrt(mean_squared_error(y_val, yp))))
            fold_mae.append(float(mean_absolute_error(y_val, yp)))

        cv_primary   = float(np.mean(fold_rmse))
        cv_secondary = float(np.mean(fold_mae))

        return {
            "primary_metric_name":  "rmse",
            "cv_primary_score":     cv_primary,
            "cv_secondary_score":   cv_secondary,
            "cv_scores_detail": {
                "fold_scores": fold_rmse,
                "fold_mae":    fold_mae,
                "mean":        cv_primary,
                "std":         float(np.std(fold_rmse)),
            },
            "model_type":       "ridge_regression",
            "duration_seconds": time.time() - t0,
        }


def evaluate_on_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    task_type: str,
    random_seed: int = 42,
) -> dict:
    """
    Test seti üzerinde nihai değerlendirme yapar.
    Strateji seçimi tamamlandıktan sonra yalnızca bir kez çağrılır.
    """
    if task_type == "classification":
        model = LogisticRegression(max_iter=1000, random_state=random_seed)
    else:
        model = Ridge()

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if task_type == "classification":
        primary   = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        secondary = float(balanced_accuracy_score(y_test, y_pred))
        return {
            "primary_metric_name":        "macro_f1",
            "final_test_primary_score":   primary,
            "final_test_secondary_score": secondary,
            "final_test_metrics_detail": {
                "macro_f1":          primary,
                "balanced_accuracy": secondary,
            },
        }
    else:
        primary   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        secondary = float(mean_absolute_error(y_test, y_pred))
        return {
            "primary_metric_name":        "rmse",
            "final_test_primary_score":   primary,
            "final_test_secondary_score": secondary,
            "final_test_metrics_detail": {
                "rmse": primary,
                "mae":  secondary,
            },
        }
