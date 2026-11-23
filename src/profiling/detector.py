"""
Sütun türü tespiti — sayısal ve kategorik ayrımı.
"""

import pandas as pd


def detect_column_types(df: pd.DataFrame, target: str) -> dict:
    """
    Veri setindeki sayısal ve kategorik sütunları belirler.
    Hedef değişkeni analiz dışında tutar.
    """

    if target not in df.columns:
        raise ValueError(f"Hedef değişken bulunamadı: {target}")

    X = df.drop(columns=[target])

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "numeric": numeric_columns,
        "categorical": categorical_columns,
    }
