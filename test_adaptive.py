from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_profile import load_dataset
from src.data_split import split_dataset
from src.preprocessing import AdaptivePreprocessor


df = load_dataset("data/raw/titanic.csv")

X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="Survived"
)

numeric_columns = ["Age", "Fare", "SibSp", "Parch"]
categorical_columns = ["Sex", "Pclass", "Embarked"]

model_b = Pipeline(steps=[
    ("preprocessor", AdaptivePreprocessor(
        numeric_columns,
        categorical_columns
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = cross_validate(
    model_b,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "macro_f1": "f1_macro"
    }
)

print("B BASELINE - 5-FOLD CV")
print("----------------------")

print("Accuracy skorları:", results["test_accuracy"])
print("Macro-F1 skorları:", results["test_macro_f1"])

print(
    f"Ortalama Accuracy: "
    f"{results['test_accuracy'].mean():.4f}"
)

print(
    f"Ortalama Macro-F1: "
    f"{results['test_macro_f1'].mean():.4f}"
)

print(
    f"Macro-F1 standart sapma: "
    f"{results['test_macro_f1'].std():.4f}"
)