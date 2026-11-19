import pandas as pd

from src.task_detection import detect_task_type
from src.data_split import split_dataset


df = pd.DataFrame({
    "area": [80, 100, 120, 150, 200, 90, 110, 140, 170, 220],
    "rooms": [2, 3, 3, 4, 5, 2, 3, 4, 4, 5],
    "price": [
        150000, 190000, 230000, 280000, 360000,
        170000, 210000, 260000, 320000, 400000
    ]
})

task_type = detect_task_type(df, "price")

X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="price",
    task_type=task_type
)

print("Problem türü:", task_type)
print("Eğitim verisi:", X_train.shape)
print("Test verisi:", X_test.shape)

print("\nEğitim hedef değerleri:")
print(y_train.values)

print("\nTest hedef değerleri:")
print(y_test.values)