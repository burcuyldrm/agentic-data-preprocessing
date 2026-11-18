"""
detect_task_type() birim testleri.

Özellikle pandas sürümüne bağlı dtype farklılıklarını kapsar:
  - object dtype (pandas < 3.0 varsayılanı)
  - StringDtype (pandas 3.0+ varsayılanı; pd.options.future.infer_string=True)
  - ArrowDtype(string) (pyarrow backend)
  - CategoricalDtype
  - bool
  - int (az / çok sınıf)
  - float (regresyon)
  - datetime / timedelta → ValueError
"""

import numpy as np
import pandas as pd
import pytest

from src.task_detection import detect_task_type


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _df(values, dtype=None, target="y"):
    s = pd.Series(values, dtype=dtype, name=target)
    return pd.DataFrame({target: s, "x": range(len(s))})


# ---------------------------------------------------------------------------
# Temel sınıflandırma
# ---------------------------------------------------------------------------

class TestClassificationDetection:
    def test_object_string_binary(self):
        df = _df(["yes", "no", "yes", "no"])
        assert detect_task_type(df, "y") == "classification"

    def test_object_string_multiclass(self):
        df = _df(["a", "b", "c", "a", "b"])
        assert detect_task_type(df, "y") == "classification"

    def test_string_dtype_binary(self):
        df = _df(["yes", "no", "yes"], dtype=pd.StringDtype())
        assert detect_task_type(df, "y") == "classification"

    def test_string_dtype_multiclass(self):
        df = _df(["cat", "dog", "bird", "cat"], dtype=pd.StringDtype())
        assert detect_task_type(df, "y") == "classification"

    def test_categorical_dtype(self):
        df = _df(["a", "b", "a"], dtype="category")
        assert detect_task_type(df, "y") == "classification"

    def test_bool_series(self):
        df = _df([True, False, True, False])
        assert detect_task_type(df, "y") == "classification"

    def test_integer_binary(self):
        df = _df([0, 1, 0, 1, 1])
        assert detect_task_type(df, "y") == "classification"

    def test_integer_binary_nonzero(self):
        df = _df([3, 7, 3, 7, 3])
        assert detect_task_type(df, "y") == "classification"

    def test_csv_roundtrip_string_target(self, tmp_path):
        """CSV'den okunan dize hedef değişken, farklı pandas sürümlerinde
        sınıflandırma olarak tespit edilmelidir."""
        rng = np.random.default_rng(42)
        path = str(tmp_path / "data.csv")
        df = pd.DataFrame({
            "x":      rng.normal(size=50),
            "target": rng.choice(["yes", "no"], size=50),
        })
        df.to_csv(path, index=False)
        df2 = pd.read_csv(path)
        assert detect_task_type(df2, "target") == "classification"

    def test_csv_roundtrip_string_target_infer_string(self, tmp_path):
        """pd.options.future.infer_string=True etkinken dize hedef
        hâlâ sınıflandırma olarak tespit edilmelidir."""
        rng = np.random.default_rng(0)
        path = str(tmp_path / "data2.csv")
        df = pd.DataFrame({
            "x":      rng.normal(size=50),
            "target": rng.choice(["yes", "no"], size=50),
        })
        df.to_csv(path, index=False)

        orig = pd.options.future.infer_string
        try:
            pd.options.future.infer_string = True
            df2 = pd.read_csv(path)
            assert detect_task_type(df2, "target") == "classification"
        finally:
            pd.options.future.infer_string = orig


# ---------------------------------------------------------------------------
# Regresyon tespiti
# ---------------------------------------------------------------------------

class TestRegressionDetection:
    def test_float_many_unique(self):
        rng = np.random.default_rng(1)
        df  = _df(rng.normal(size=100).tolist())
        assert detect_task_type(df, "y") == "regression"

    def test_integer_many_unique(self):
        df = _df(list(range(50)))
        assert detect_task_type(df, "y") == "regression"

    def test_csv_roundtrip_float_target(self, tmp_path):
        rng = np.random.default_rng(2)
        path = str(tmp_path / "reg.csv")
        df = pd.DataFrame({
            "x":      rng.normal(size=100),
            "target": rng.normal(size=100),
        })
        df.to_csv(path, index=False)
        df2 = pd.read_csv(path)
        assert detect_task_type(df2, "target") == "regression"


# ---------------------------------------------------------------------------
# Hata durumları
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_target_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Hedef sütun bulunamadı"):
            detect_task_type(df, "y")

    def test_all_nan_target(self):
        df = _df([None, None, None])
        with pytest.raises(ValueError, match="geçerli değer yok"):
            detect_task_type(df, "y")

    def test_datetime_target_raises(self):
        df = _df(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        with pytest.raises(ValueError, match="Desteklenmeyen"):
            detect_task_type(df, "y")

    def test_timedelta_target_raises(self):
        df = _df(pd.to_timedelta(["1 days", "2 days", "3 days"]))
        with pytest.raises(ValueError, match="Desteklenmeyen"):
            detect_task_type(df, "y")


# ---------------------------------------------------------------------------
# ArrowDtype (isteğe bağlı; pyarrow yüklü değilse atlanır)
# ---------------------------------------------------------------------------

class TestArrowDtype:
    def test_arrow_string_classification(self):
        pytest.importorskip("pyarrow")
        s = pd.Series(["a", "b", "a"], dtype="string[pyarrow]")
        df = pd.DataFrame({"y": s, "x": range(3)})
        assert detect_task_type(df, "y") == "classification"
