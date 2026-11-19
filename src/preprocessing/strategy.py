"""
Veri dağılımına göre sayısal ön işleme strateji seçimi.
"""

import pandas as pd


def select_preprocessing_strategy(
    X_train: pd.DataFrame,
    numeric_columns: list
) -> dict:
    """
    Eğitim verisinin dağılımına göre sayısal
    ön işleme stratejilerini belirler.
    """

    strategies = {}

    for column in numeric_columns:
        skewness = X_train[column].skew()

        if pd.isna(skewness):
            imputer = "median"
            scaler = "standard"

        elif abs(skewness) >= 1:
            imputer = "median"
            scaler = "robust"

        else:
            imputer = "mean"
            scaler = "standard"

        strategies[column] = {
            "skewness": (
                None if pd.isna(skewness)
                else round(float(skewness), 4)
            ),
            "imputer": imputer,
            "scaler": scaler
        }

    return strategies
