from src.data_profile import load_dataset
from src.data_split import split_dataset
from src.preprocessing import build_baseline_preprocessor

df = load_dataset("data/raw/titanic.csv")

X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="Survived"
)

# İlk denemede kontrollü bir özellik kümesi kullanıyoruz.
numeric_columns = ["Age", "Fare", "SibSp", "Parch"]
categorical_columns = ["Sex", "Pclass", "Embarked"]

preprocessor = build_baseline_preprocessor(
    numeric_columns,
    categorical_columns
)

# Ön işleme kurallarını yalnızca eğitim verisinden öğren.
X_train_processed = preprocessor.fit_transform(X_train)

# Aynı kuralları test verisine uygula.
X_test_processed = preprocessor.transform(X_test)

print("İşlenmiş eğitim verisi:", X_train_processed.shape)
print("İşlenmiş test verisi:", X_test_processed.shape)

print("Eğitim verisinde NaN var mı?:",
      __import__("numpy").isnan(X_train_processed).any())

print("Test verisinde NaN var mı?:",
      __import__("numpy").isnan(X_test_processed).any())