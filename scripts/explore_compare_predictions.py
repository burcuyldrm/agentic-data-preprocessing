import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

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

model_a = Pipeline([
    ("preprocessor", build_baseline_preprocessor(
        numeric_columns, categorical_columns
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

model_b = Pipeline([
    ("preprocessor", AdaptivePreprocessor(
        numeric_columns, categorical_columns
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

model_a.fit(X_train, y_train)
model_b.fit(X_train, y_train)

pred_a = model_a.predict(X_test)
pred_b = model_b.predict(X_test)

different = np.sum(pred_a != pred_b)

print("A ve B tahmin karşılaştırması")
print("-----------------------------")
print("Test gözlem sayısı:", len(y_test))
print("Farklı tahmin sayısı:", different)
print("A Accuracy:", round(accuracy_score(y_test, pred_a), 4))
print("B Accuracy:", round(accuracy_score(y_test, pred_b), 4))
print("A Macro-F1:", round(f1_score(y_test, pred_a, average="macro"), 4))
print("B Macro-F1:", round(f1_score(y_test, pred_b, average="macro"), 4))