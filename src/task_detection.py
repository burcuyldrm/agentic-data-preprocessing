import pandas as pd


def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """
    Hedef değişkenin türüne göre
    sınıflandırma veya regresyon belirler.
    """

    if target not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target}")

    y = df[target].dropna()

    if y.empty:
        raise ValueError("Hedef değişkende geçerli değer yok.")

    if (
        pd.api.types.is_object_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
    ):
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        unique_count = y.nunique()

        if unique_count <= 2:
            return "classification"

        return "regression"

    raise ValueError("Desteklenmeyen hedef değişken türü.")