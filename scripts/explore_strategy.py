from src.data_profile import load_dataset
from src.data_split import split_dataset
from src.strategy_selector import select_preprocessing_strategy

df = load_dataset("data/raw/titanic.csv")

X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="Survived"
)

numeric_columns = ["Age", "Fare", "SibSp", "Parch"]

strategies = select_preprocessing_strategy(
    X_train,
    numeric_columns
)

print("B BASELINE - ÖN İŞLEME STRATEJİLERİ")
print("----------------------------------")

for column, strategy in strategies.items():
    print(f"\nSütun: {column}")
    print(f"Çarpıklık: {strategy['skewness']}")
    print(f"Eksik değer yöntemi: {strategy['imputer']}")
    print(f"Ölçekleme yöntemi: {strategy['scaler']}")