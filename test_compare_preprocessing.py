import numpy as np

from src.data_profile import load_dataset
from src.data_split import split_dataset
from src.preprocessing import (
    build_baseline_preprocessor,
    AdaptivePreprocessor
)

df = load_dataset("data/raw/titanic.csv")

X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="Survived"
)

numeric_columns = ["Age", "Fare", "SibSp", "Parch"]
categorical_columns = ["Sex", "Pclass", "Embarked"]

# A yaklaşımı
preprocessor_a = build_baseline_preprocessor(
    numeric_columns,
    categorical_columns
)

X_a = preprocessor_a.fit_transform(X_train)

# B yaklaşımı
preprocessor_b = AdaptivePreprocessor(
    numeric_columns,
    categorical_columns
)

X_b = preprocessor_b.fit_transform(X_train)

print("A veri boyutu:", X_a.shape)
print("B veri boyutu:", X_b.shape)

print("\nDönüştürülmüş veriler tamamen aynı mı?")
print(np.allclose(X_a, X_b))

print("\nİlk gözlemin sayısal özellikleri:")
print("A:", X_a[0, :4])
print("B:", X_b[0, :4])

print("\nB tarafından seçilen stratejiler:")
for column, strategy in preprocessor_b.strategies_.items():
    print(column, strategy)