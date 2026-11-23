"""
Veri yükleme — CSV okuma ve temel doğrulama.
"""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    CSV dosyasını okuyarak pandas DataFrame döndürür.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dosya bulunamadı: {file_path}"
        )

    if path.suffix.lower() != ".csv":
        raise ValueError("Yalnızca CSV dosyaları destekleniyor.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Veri seti boş.")

    return df
