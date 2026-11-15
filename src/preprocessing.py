from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from src.strategy_selector import select_preprocessing_strategy


def build_baseline_preprocessor(numeric_columns, categorical_columns):
    """
    Sabit ön işleme (A baseline) yapısını oluşturur.
    """

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )

    return preprocessor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from src.strategy_selector import select_preprocessing_strategy


class AdaptivePreprocessor(BaseEstimator, TransformerMixin):
    """
    Eğitim verisinin dağılımına göre ön işleme
    stratejilerini seçen dönüştürücü.
    """

    def __init__(self, numeric_columns, categorical_columns):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        # Stratejiler yalnızca fit sırasında öğrenilir.
        self.strategies_ = select_preprocessing_strategy(
            X, self.numeric_columns
        )

        transformers = []

        for column, strategy in self.strategies_.items():
            scaler = (
                RobustScaler()
                if strategy["scaler"] == "robust"
                else StandardScaler()
            )

            numeric_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(
                    strategy=strategy["imputer"]
                )),
                ("scaler", scaler)
            ])

            transformers.append((
                f"numeric_{column}",
                numeric_pipeline,
                [column]
            ))

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            self.categorical_columns
        ))

        self.preprocessor_ = ColumnTransformer(
            transformers=transformers
        )

        self.preprocessor_.fit(X, y)
        return self

    def transform(self, X):
        return self.preprocessor_.transform(X)