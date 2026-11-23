"""
src.profiling alt modüllerinin import yolu ve kimlik testleri.

Eski yolların (src.data_profile) ve yeni yolların (src.profiling.*)
aynı fonksiyon nesnelerini gösterdiğini doğrular.
"""


class TestDirectImports:
    def test_loader_importable(self):
        from src.profiling.loader import load_dataset
        assert callable(load_dataset)

    def test_detector_importable(self):
        from src.profiling.detector import detect_column_types
        assert callable(detect_column_types)

    def test_profiler_importable(self):
        from src.profiling.profiler import (
            analyze_distribution,
            analyze_missing_values,
            analyze_outliers,
            generate_data_profile,
        )
        assert all(callable(f) for f in [
            analyze_distribution, analyze_missing_values,
            analyze_outliers, generate_data_profile,
        ])

    def test_package_init_importable(self):
        from src.profiling import (
            analyze_distribution,
            analyze_missing_values,
            analyze_outliers,
            detect_column_types,
            generate_data_profile,
            load_dataset,
        )
        assert all(x is not None for x in [
            load_dataset, detect_column_types, analyze_missing_values,
            analyze_distribution, analyze_outliers, generate_data_profile,
        ])


class TestShimIdentity:
    def test_load_dataset_same_object(self):
        from src.data_profile import load_dataset as Old
        from src.profiling.loader import load_dataset as New
        assert Old is New

    def test_detect_column_types_same_object(self):
        from src.data_profile import detect_column_types as Old
        from src.profiling.detector import detect_column_types as New
        assert Old is New

    def test_analyze_missing_values_same_object(self):
        from src.data_profile import analyze_missing_values as Old
        from src.profiling.profiler import analyze_missing_values as New
        assert Old is New

    def test_analyze_distribution_same_object(self):
        from src.data_profile import analyze_distribution as Old
        from src.profiling.profiler import analyze_distribution as New
        assert Old is New

    def test_analyze_outliers_same_object(self):
        from src.data_profile import analyze_outliers as Old
        from src.profiling.profiler import analyze_outliers as New
        assert Old is New

    def test_generate_data_profile_same_object(self):
        from src.data_profile import generate_data_profile as Old
        from src.profiling.profiler import generate_data_profile as New
        assert Old is New

    def test_package_init_same_as_submodules(self):
        from src.profiling import load_dataset as PkgLoad
        from src.profiling.loader import load_dataset as ModLoad
        assert PkgLoad is ModLoad
