import pandas as pd


def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """
    Hedef değişkenin türüne göre sınıflandırma veya regresyon belirler.

    Pandas 3.0+ ile gelen StringDtype ve ArrowDtype(string) gibi
    yeni dize türleri de sınıflandırma olarak kabul edilir.
    Yalnızca datetime/timedelta türleri desteklenmez.
    """

    if target not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target}")

    y = df[target].dropna()

    if y.empty:
        raise ValueError("Hedef değişkende geçerli değer yok.")

    # Gerçekten desteklenmeyen türler: zaman damgası ve süre.
    if (
        pd.api.types.is_datetime64_any_dtype(y)
        or pd.api.types.is_timedelta64_dtype(y)
    ):
        raise ValueError("Desteklenmeyen hedef değişken türü.")

    # Sayısal türler: benzersiz değer sayısına göre karar ver.
    # bool pandas'ta numeric_dtype sayılır; 2 benzersiz değer → sınıflandırma.
    if pd.api.types.is_numeric_dtype(y):
        unique_count = y.nunique()

        if unique_count <= 2:
            return "classification"

        return "regression"

    # Sayısal olmayan tüm türler sınıflandırma kabul edilir:
    #   object (pandas < 3.0), StringDtype (pandas 3.0+),
    #   ArrowDtype(string), CategoricalDtype, bool (object-backed).
    return "classification"