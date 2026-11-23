"""
Veri bölme — eğitim/test kümesi ayrımı.
"""

from sklearn.model_selection import train_test_split


def split_dataset(
    df,
    target,
    task_type="classification",
    test_size=0.2,
    random_state=42,
):
    """
    Veri setini eğitim ve test kümelerine ayırır.

    Classification: sınıf oranları korunur.
    Regression: normal rastgele bölme yapılır.
    """

    if target not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target}")

    if task_type not in ["classification", "regression"]:
        raise ValueError("Geçersiz problem türü.")

    X = df.drop(columns=[target])
    y = df[target]

    if y.isna().any():
        raise ValueError("Hedef değişkende eksik değer bulunuyor.")

    stratify = y if task_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test
