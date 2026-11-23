"""
src.datasets.split modülünün import yolu ve kimlik testleri.

Üç import yolunun aynı fonksiyon nesnesini gösterdiğini doğrular:
  - src.data_split.split_dataset
  - src.datasets.split.split_dataset
  - src.datasets.split_dataset
"""


class TestDirectImports:
    def test_split_module_importable(self):
        from src.datasets.split import split_dataset
        assert callable(split_dataset)

    def test_package_init_importable(self):
        from src.datasets import split_dataset
        assert callable(split_dataset)


class TestSplitPathIdentity:
    def test_shim_same_as_new_module(self):
        from src.data_split import split_dataset as Old
        from src.datasets.split import split_dataset as New
        assert Old is New

    def test_package_init_same_as_new_module(self):
        from src.datasets import split_dataset as PkgSplit
        from src.datasets.split import split_dataset as ModSplit
        assert PkgSplit is ModSplit

    def test_shim_same_as_package_init(self):
        from src.data_split import split_dataset as Old
        from src.datasets import split_dataset as PkgSplit
        assert Old is PkgSplit

    def test_runner_uses_new_path(self):
        """runner.py artık src.datasets.split'ten içe aktarıyor."""
        from src.datasets.split import split_dataset as DirectSplit
        from src.orchestration import runner as _runner_mod
        import inspect
        src = inspect.getfile(_runner_mod)
        # runner modülünde src.datasets.split'ten gelen nesneyi doğrula
        assert DirectSplit is _runner_mod.__dict__.get(
            "split_dataset", DirectSplit
        )
