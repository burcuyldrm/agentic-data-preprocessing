from src.data_profile import load_dataset
from src.data_split import split_dataset

df = load_dataset("data/raw/titanic.csv")

X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="Survived"
)

print("Eğitim verisi:", X_train.shape)
print("Test verisi:", X_test.shape)

print("\nEğitim hedef dağılımı:")
print(y_train.value_counts(normalize=True))

print("\nTest hedef dağılımı:")
print(y_test.value_counts(normalize=True))