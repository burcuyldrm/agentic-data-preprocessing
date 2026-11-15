from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_profile import load_dataset
from src.data_split import split_dataset
from src.preprocessing import build_baseline_preprocessor


# 1. Veri setini yükle
df = load_dataset("data/raw/titanic.csv")

# 2. Eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="Survived"
)

# 3. Kullanılacak özellikleri belirle
numeric_columns = ["Age", "Fare", "SibSp", "Parch"]
categorical_columns = ["Sex", "Pclass", "Embarked"]

# 4. Ön işleme yapısını oluştur
preprocessor = build_baseline_preprocessor(
    numeric_columns,
    categorical_columns
)

# 5. Ön işleme ve modeli birleştir
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 6. Modeli eğit
model.fit(X_train, y_train)

# 7. Test verisinde tahmin yap
y_pred = model.predict(X_test)

# 8. Başarı metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("A BASELINE SONUÇLARI")
print("--------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro-F1: {f1:.4f}")

# 9. Beş katlı çapraz doğrulama
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "macro_f1": "f1_macro"
    }
)

print("\n5-FOLD CROSS VALIDATION")
print("-----------------------")

print("Accuracy skorları:", cv_results["test_accuracy"])
print("Macro-F1 skorları:", cv_results["test_macro_f1"])

print(
    f"Ortalama Accuracy: "
    f"{cv_results['test_accuracy'].mean():.4f}"
)

print(
    f"Ortalama Macro-F1: "
    f"{cv_results['test_macro_f1'].mean():.4f}"
)

print(
    f"Macro-F1 standart sapma: "
    f"{cv_results['test_macro_f1'].std():.4f}"
)