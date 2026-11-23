"""
Veri profilleme — eksik değer, dağılım, aykırı değer ve profil oluşturma.
"""

import pandas as pd

from src.profiling.detector import detect_column_types


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her sütundaki eksik değer sayısını ve oranını hesaplar.
    """

    missing_count = df.isnull().sum()

    missing_percent = (missing_count / len(df)) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent,
    })

    return missing_report


def analyze_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Sayısal özelliklerin dağılımını ve çarpıklığını analiz eder.
    """

    X = df.drop(columns=[target])

    numeric_df = X.select_dtypes(include=["number"])

    report = pd.DataFrame({
        "mean":     numeric_df.mean(),
        "median":   numeric_df.median(),
        "std":      numeric_df.std(),
        "skewness": numeric_df.skew(),
    })

    return report


def analyze_outliers(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Sayısal sütunlardaki potansiyel aykırı değerleri
    IQR yöntemiyle analiz eder.
    """

    X = df.drop(columns=[target])
    numeric_df = X.select_dtypes(include=["number"])

    results = []

    for column in numeric_df.columns:
        values = numeric_df[column].dropna()

        if len(values) < 4:
            results.append({
                "column":          column,
                "outlier_count":   None,
                "outlier_percent": None,
            })
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (values < lower_bound) |
            (values > upper_bound)
        )

        results.append({
            "column":          column,
            "outlier_count":   int(outliers.sum()),
            "outlier_percent": round(outliers.mean() * 100, 2),
        })

    return pd.DataFrame(results)


def generate_data_profile(df: pd.DataFrame, target: str) -> dict:
    """
    Veri setinin temel profilini oluşturur.
    """

    if target not in df.columns:
        raise ValueError(f"Hedef değişken bulunamadı: {target}")

    column_types  = detect_column_types(df, target)
    missing_values = analyze_missing_values(df)
    distribution   = analyze_distribution(df, target)
    outliers       = analyze_outliers(df, target)

    profile = {
        "dataset_info": {
            "rows":    len(df),
            "columns": len(df.columns),
            "target":  target,
        },
        "column_types":   column_types,
        "missing_values": missing_values.to_dict(orient="index"),
        "distribution":   distribution.to_dict(orient="index"),
        "outliers":       outliers.to_dict(orient="records"),
    }

    return profile
