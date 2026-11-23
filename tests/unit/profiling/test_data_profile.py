"""
Karakterizasyon testleri: src.data_profile içindeki 6 fonksiyonun
mevcut davranışını belgeler ve ayrıştırma sonrası regresyonu önler.

Import yolu: src.data_profile (shim sonrası da çalışır).
"""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_returns_dataframe(self, tmp_path):
        from src.data_profile import load_dataset
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n3,4\n")
        result = load_dataset(str(f))
        assert isinstance(result, pd.DataFrame)

    def test_shape_matches_csv(self, tmp_path):
        from src.data_profile import load_dataset
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n")
        df = load_dataset(str(f))
        assert df.shape == (3, 3)

    def test_column_names_preserved(self, tmp_path):
        from src.data_profile import load_dataset
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,x\n2,y\n")
        df = load_dataset(str(f))
        assert list(df.columns) == ["col1", "col2"]

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from src.data_profile import load_dataset
        with pytest.raises(FileNotFoundError):
            load_dataset(str(tmp_path / "nonexistent.csv"))

    def test_non_csv_extension_raises_value_error(self, tmp_path):
        from src.data_profile import load_dataset
        f = tmp_path / "data.txt"
        f.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="CSV"):
            load_dataset(str(f))

    def test_empty_csv_raises_value_error(self, tmp_path):
        from src.data_profile import load_dataset
        f = tmp_path / "empty.csv"
        f.write_text("a,b\n")
        with pytest.raises(ValueError, match="boş"):
            load_dataset(str(f))

    def test_csv_values_readable(self, tmp_path):
        from src.data_profile import load_dataset
        f = tmp_path / "data.csv"
        f.write_text("x\n42\n")
        df = load_dataset(str(f))
        assert df["x"].iloc[0] == 42


# ---------------------------------------------------------------------------
# detect_column_types
# ---------------------------------------------------------------------------

class TestDetectColumnTypes:
    def _mixed_df(self):
        return pd.DataFrame({
            "num1":   [1.0, 2.0, 3.0],
            "num2":   [4,   5,   6  ],
            "cat1":   ["a", "b", "c"],
            "target": [0,   1,   0  ],
        })

    def test_returns_dict_with_expected_keys(self):
        from src.data_profile import detect_column_types
        result = detect_column_types(self._mixed_df(), "target")
        assert set(result.keys()) == {"numeric", "categorical"}

    def test_numeric_columns_identified(self):
        from src.data_profile import detect_column_types
        result = detect_column_types(self._mixed_df(), "target")
        assert "num1" in result["numeric"]
        assert "num2" in result["numeric"]

    def test_categorical_columns_identified(self):
        from src.data_profile import detect_column_types
        result = detect_column_types(self._mixed_df(), "target")
        assert "cat1" in result["categorical"]

    def test_target_excluded_from_numeric(self):
        from src.data_profile import detect_column_types
        result = detect_column_types(self._mixed_df(), "target")
        assert "target" not in result["numeric"]

    def test_target_excluded_from_categorical(self):
        from src.data_profile import detect_column_types
        result = detect_column_types(self._mixed_df(), "target")
        assert "target" not in result["categorical"]

    def test_missing_target_raises_value_error(self):
        from src.data_profile import detect_column_types
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            detect_column_types(df, "missing_target")

    def test_bool_column_in_categorical(self):
        from src.data_profile import detect_column_types
        df = pd.DataFrame({
            "flag":   pd.array([True, False, True], dtype="bool"),
            "num":    [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        result = detect_column_types(df, "target")
        assert "flag" in result["categorical"]

    def test_category_dtype_in_categorical(self):
        from src.data_profile import detect_column_types
        df = pd.DataFrame({
            "cat":    pd.Categorical(["a", "b", "a"]),
            "target": [0, 1, 0],
        })
        result = detect_column_types(df, "target")
        assert "cat" in result["categorical"]

    def test_returns_lists_not_index(self):
        from src.data_profile import detect_column_types
        result = detect_column_types(self._mixed_df(), "target")
        assert isinstance(result["numeric"], list)
        assert isinstance(result["categorical"], list)

    def test_all_numeric_df(self):
        from src.data_profile import detect_column_types
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "target": [0, 1]})
        result = detect_column_types(df, "target")
        assert len(result["categorical"]) == 0
        assert len(result["numeric"]) == 2


# ---------------------------------------------------------------------------
# analyze_missing_values
# ---------------------------------------------------------------------------

class TestAnalyzeMissingValues:
    def test_returns_dataframe(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = analyze_missing_values(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_missing_count_column(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"a": [1, 2]})
        result = analyze_missing_values(df)
        assert "missing_count" in result.columns

    def test_has_missing_percent_column(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"a": [1, 2]})
        result = analyze_missing_values(df)
        assert "missing_percent" in result.columns

    def test_no_missing_all_zero(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = analyze_missing_values(df)
        assert result["missing_count"].sum() == 0
        assert result["missing_percent"].sum() == 0.0

    def test_partial_missing_count(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
        result = analyze_missing_values(df)
        assert result.loc["a", "missing_count"] == 2

    def test_partial_missing_percent(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"a": [1, None, 3, None], "b": [1, 2, 3, 4]})
        result = analyze_missing_values(df)
        assert result.loc["a", "missing_percent"] == pytest.approx(50.0)

    def test_fully_missing_column(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
        result = analyze_missing_values(df)
        assert result.loc["a", "missing_count"] == 3
        assert result.loc["a", "missing_percent"] == pytest.approx(100.0)

    def test_index_is_column_names(self):
        from src.data_profile import analyze_missing_values
        df = pd.DataFrame({"x": [1, 2], "y": [3, None]})
        result = analyze_missing_values(df)
        assert set(result.index) == {"x", "y"}


# ---------------------------------------------------------------------------
# analyze_distribution
# ---------------------------------------------------------------------------

class TestAnalyzeDistribution:
    def _df(self):
        return pd.DataFrame({
            "num":    [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat":    ["a", "b", "a", "b", "a"],
            "target": [0,   1,   0,   1,   0  ],
        })

    def test_returns_dataframe(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert isinstance(result, pd.DataFrame)

    def test_has_mean_column(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert "mean" in result.columns

    def test_has_median_column(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert "median" in result.columns

    def test_has_std_column(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert "std" in result.columns

    def test_has_skewness_column(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert "skewness" in result.columns

    def test_target_excluded(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert "target" not in result.index

    def test_non_numeric_excluded(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert "cat" not in result.index

    def test_mean_correct(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert result.loc["num", "mean"] == pytest.approx(3.0)

    def test_median_correct(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert result.loc["num", "median"] == pytest.approx(3.0)

    def test_skewness_near_zero_for_symmetric(self):
        from src.data_profile import analyze_distribution
        result = analyze_distribution(self._df(), "target")
        assert abs(result.loc["num", "skewness"]) < 0.1

    def test_no_numeric_columns_returns_empty_df(self):
        from src.data_profile import analyze_distribution
        df = pd.DataFrame({"cat": ["a", "b", "c"], "target": [0, 1, 0]})
        result = analyze_distribution(df, "target")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# analyze_outliers
# ---------------------------------------------------------------------------

class TestAnalyzeOutliers:
    def _df_no_outliers(self):
        return pd.DataFrame({
            "num":    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "target": [0] * 10,
        })

    def _df_with_outlier(self):
        return pd.DataFrame({
            "num":    [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "target": [0] * 6,
        })

    def test_returns_dataframe(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_no_outliers(), "target")
        assert isinstance(result, pd.DataFrame)

    def test_has_column_field(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_no_outliers(), "target")
        assert "column" in result.columns

    def test_has_outlier_count_field(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_no_outliers(), "target")
        assert "outlier_count" in result.columns

    def test_has_outlier_percent_field(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_no_outliers(), "target")
        assert "outlier_percent" in result.columns

    def test_no_outliers_zero_count(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_no_outliers(), "target")
        row = result[result["column"] == "num"].iloc[0]
        assert row["outlier_count"] == 0

    def test_outlier_detected(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_with_outlier(), "target")
        row = result[result["column"] == "num"].iloc[0]
        assert row["outlier_count"] >= 1

    def test_outlier_percent_is_float(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_with_outlier(), "target")
        row = result[result["column"] == "num"].iloc[0]
        assert isinstance(row["outlier_percent"], float)

    def test_fewer_than_4_values_count_is_none(self):
        from src.data_profile import analyze_outliers
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        result = analyze_outliers(df, "target")
        row = result[result["column"] == "num"].iloc[0]
        assert row["outlier_count"] is None

    def test_fewer_than_4_values_percent_is_none(self):
        from src.data_profile import analyze_outliers
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        result = analyze_outliers(df, "target")
        row = result[result["column"] == "num"].iloc[0]
        assert row["outlier_percent"] is None

    def test_target_excluded(self):
        from src.data_profile import analyze_outliers
        result = analyze_outliers(self._df_no_outliers(), "target")
        assert "target" not in result["column"].values

    def test_all_nan_column_treated_as_fewer_than_4(self):
        from src.data_profile import analyze_outliers
        df = pd.DataFrame({
            "num":    [float("nan")] * 5,
            "target": [0] * 5,
        })
        result = analyze_outliers(df, "target")
        row = result[result["column"] == "num"].iloc[0]
        assert row["outlier_count"] is None


# ---------------------------------------------------------------------------
# generate_data_profile
# ---------------------------------------------------------------------------

class TestGenerateDataProfile:
    def _df(self):
        return pd.DataFrame({
            "num1":   [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat1":   ["a", "b", "a", "b", "a"],
            "target": [0,   1,   0,   1,   0  ],
        })

    def test_returns_dict(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert isinstance(result, dict)

    def test_has_dataset_info_key(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert "dataset_info" in result

    def test_has_column_types_key(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert "column_types" in result

    def test_has_missing_values_key(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert "missing_values" in result

    def test_has_distribution_key(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert "distribution" in result

    def test_has_outliers_key(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert "outliers" in result

    def test_dataset_info_rows(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert result["dataset_info"]["rows"] == 5

    def test_dataset_info_columns(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert result["dataset_info"]["columns"] == 3

    def test_dataset_info_target(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert result["dataset_info"]["target"] == "target"

    def test_missing_target_raises_value_error(self):
        from src.data_profile import generate_data_profile
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError):
            generate_data_profile(df, "missing_target")

    def test_outliers_is_list(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert isinstance(result["outliers"], list)

    def test_missing_values_is_dict(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert isinstance(result["missing_values"], dict)

    def test_distribution_is_dict(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert isinstance(result["distribution"], dict)

    def test_column_types_has_numeric_key(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert "numeric" in result["column_types"]

    def test_column_types_has_categorical_key(self):
        from src.data_profile import generate_data_profile
        result = generate_data_profile(self._df(), "target")
        assert "categorical" in result["column_types"]
