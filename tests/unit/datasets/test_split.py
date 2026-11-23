"""
Karakterizasyon testleri: src.data_split.split_dataset fonksiyonunun
mevcut davranışını belgeler ve ayrıştırma sonrası regresyonu önler.

Import yolu: src.data_split (shim sonrası da çalışır).
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Yardımcı veri üreticileri
# ---------------------------------------------------------------------------

def _clf_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat1":  rng.normal(size=n),
        "feat2":  rng.integers(0, 10, size=n).astype(float),
        "target": ["a"] * (n // 2) + ["b"] * (n // 2),
    })


def _reg_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat1":  rng.normal(size=n),
        "feat2":  rng.integers(0, 10, size=n).astype(float),
        "target": rng.normal(size=n),
    })


# ---------------------------------------------------------------------------
# Temel dönüş değerleri
# ---------------------------------------------------------------------------

class TestSplitDatasetReturnValues:
    def test_returns_four_values(self):
        from src.data_split import split_dataset
        result = split_dataset(_clf_df(), "target")
        assert len(result) == 4

    def test_x_train_is_dataframe(self):
        from src.data_split import split_dataset
        X_train, _, _, _ = split_dataset(_clf_df(), "target")
        assert isinstance(X_train, pd.DataFrame)

    def test_x_test_is_dataframe(self):
        from src.data_split import split_dataset
        _, X_test, _, _ = split_dataset(_clf_df(), "target")
        assert isinstance(X_test, pd.DataFrame)

    def test_y_train_is_series(self):
        from src.data_split import split_dataset
        _, _, y_train, _ = split_dataset(_clf_df(), "target")
        assert isinstance(y_train, pd.Series)

    def test_y_test_is_series(self):
        from src.data_split import split_dataset
        _, _, _, y_test = split_dataset(_clf_df(), "target")
        assert isinstance(y_test, pd.Series)

    def test_target_excluded_from_x_train(self):
        from src.data_split import split_dataset
        X_train, _, _, _ = split_dataset(_clf_df(), "target")
        assert "target" not in X_train.columns

    def test_target_excluded_from_x_test(self):
        from src.data_split import split_dataset
        _, X_test, _, _ = split_dataset(_clf_df(), "target")
        assert "target" not in X_test.columns

    def test_total_rows_preserved(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        X_train, X_test, _, _ = split_dataset(df, "target")
        assert len(X_train) + len(X_test) == 20


# ---------------------------------------------------------------------------
# Varsayılan parametreler
# ---------------------------------------------------------------------------

class TestSplitDatasetDefaultParameters:
    def test_default_test_size_is_20_percent(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        _, X_test, _, _ = split_dataset(df, "target")
        assert len(X_test) == 4  # 20 * 0.2

    def test_default_train_size_is_80_percent(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        X_train, _, _, _ = split_dataset(df, "target")
        assert len(X_train) == 16  # 20 * 0.8

    def test_default_random_state_matches_42(self):
        """Varsayılan random_state=42 ile açık 42 aynı sonucu verir."""
        from src.data_split import split_dataset
        df = _reg_df(n=20)
        X_default, _, _, _ = split_dataset(df, "target", task_type="regression")
        X_explicit, _, _, _ = split_dataset(df, "target", task_type="regression", random_state=42)
        assert X_default.equals(X_explicit)

    def test_default_task_type_is_classification(self):
        """Varsayılan task_type=classification ile stratification uygulanır."""
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        # Classification default'ta stratify=y kullanılır; sınıf oranları korunur
        _, X_test, _, y_test = split_dataset(df, "target")
        assert set(y_test.unique()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

class TestSplitDatasetStratification:
    def test_classification_preserves_class_proportions_train(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)  # 10a + 10b, %50/%50
        X_train, _, y_train, _ = split_dataset(df, "target", task_type="classification")
        a_pct = (y_train == "a").mean()
        assert abs(a_pct - 0.5) < 0.01

    def test_classification_preserves_class_proportions_test(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        _, X_test, _, y_test = split_dataset(df, "target", task_type="classification")
        a_pct = (y_test == "a").mean()
        assert abs(a_pct - 0.5) < 0.01

    def test_regression_does_not_require_stratification(self):
        from src.data_split import split_dataset
        df = _reg_df(n=20)
        X_train, X_test, _, _ = split_dataset(df, "target", task_type="regression")
        assert len(X_train) + len(X_test) == 20

    def test_single_class_target_does_not_raise(self):
        """Tüm etiketler aynı sınıfsa stratification hata vermez."""
        from src.data_split import split_dataset
        df = pd.DataFrame({"feat": range(10), "target": [0] * 10})
        X_train, X_test, _, _ = split_dataset(df, "target", task_type="classification")
        assert len(X_train) + len(X_test) == 10

    def test_low_count_class_stratification_raises(self):
        """Yalnızca 1 üyeli sınıf stratification'da ValueError üretir."""
        from src.data_split import split_dataset
        df = pd.DataFrame({
            "feat":   range(10),
            "target": [0] * 9 + [1],  # sınıf "1" yalnızca 1 üyeye sahip
        })
        with pytest.raises(ValueError):
            split_dataset(df, "target", task_type="classification")


# ---------------------------------------------------------------------------
# Tekrarlanabilirlik
# ---------------------------------------------------------------------------

class TestSplitDatasetReproducibility:
    def test_same_seed_produces_same_split(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        X1, _, _, _ = split_dataset(df, "target", task_type="classification", random_state=7)
        X2, _, _, _ = split_dataset(df, "target", task_type="classification", random_state=7)
        assert X1.equals(X2)

    def test_different_seeds_produce_different_splits(self):
        from src.data_split import split_dataset
        df = _reg_df(n=20)
        X1, _, _, _ = split_dataset(df, "target", task_type="regression", random_state=0)
        X2, _, _, _ = split_dataset(df, "target", task_type="regression", random_state=99)
        assert not X1.equals(X2)

    def test_custom_test_size_applied(self):
        from src.data_split import split_dataset
        df = _reg_df(n=20)
        _, X_test, _, _ = split_dataset(df, "target", task_type="regression", test_size=0.4)
        assert len(X_test) == 8  # 20 * 0.4


# ---------------------------------------------------------------------------
# Veri bütünlüğü
# ---------------------------------------------------------------------------

class TestSplitDatasetDataIntegrity:
    def test_input_dataframe_not_modified(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        df_copy = df.copy()
        split_dataset(df, "target")
        assert df.equals(df_copy)

    def test_x_y_index_aligned_train(self):
        from src.data_split import split_dataset
        X_train, _, y_train, _ = split_dataset(_clf_df(n=20), "target")
        assert list(X_train.index) == list(y_train.index)

    def test_x_y_index_aligned_test(self):
        from src.data_split import split_dataset
        _, X_test, _, y_test = split_dataset(_clf_df(n=20), "target")
        assert list(X_test.index) == list(y_test.index)

    def test_train_test_indices_disjoint(self):
        from src.data_split import split_dataset
        X_train, X_test, _, _ = split_dataset(_clf_df(n=20), "target")
        assert set(X_train.index).isdisjoint(set(X_test.index))

    def test_train_test_indices_cover_all_rows(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        X_train, X_test, _, _ = split_dataset(df, "target")
        all_idx = set(X_train.index) | set(X_test.index)
        assert all_idx == set(df.index)

    def test_feature_columns_preserved(self):
        from src.data_split import split_dataset
        df = _clf_df(n=20)
        X_train, _, _, _ = split_dataset(df, "target")
        assert set(X_train.columns) == {"feat1", "feat2"}


# ---------------------------------------------------------------------------
# Hata senaryoları
# ---------------------------------------------------------------------------

class TestSplitDatasetErrors:
    def test_missing_target_raises_value_error(self):
        from src.data_split import split_dataset
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            split_dataset(df, "missing_col")

    def test_missing_target_error_message(self):
        from src.data_split import split_dataset
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="Hedef sütun bulunamadı"):
            split_dataset(df, "missing_col")

    def test_nan_in_target_raises_value_error(self):
        from src.data_split import split_dataset
        df = pd.DataFrame({"feat": [1, 2, 3, 4], "target": [1.0, None, 3.0, 4.0]})
        with pytest.raises(ValueError, match="eksik"):
            split_dataset(df, "target")

    def test_invalid_task_type_raises_value_error(self):
        from src.data_split import split_dataset
        df = _clf_df(n=10)
        with pytest.raises(ValueError):
            split_dataset(df, "target", task_type="clustering")

    def test_invalid_task_type_error_message(self):
        from src.data_split import split_dataset
        df = _clf_df(n=10)
        with pytest.raises(ValueError, match="Geçersiz problem türü"):
            split_dataset(df, "target", task_type="unknown")

    def test_very_small_dataset_raises(self):
        """1 satırlık DataFrame bölme sırasında ValueError üretir."""
        from src.data_split import split_dataset
        df = pd.DataFrame({"feat": [1.0], "target": [0]})
        with pytest.raises(ValueError):
            split_dataset(df, "target", task_type="regression")
