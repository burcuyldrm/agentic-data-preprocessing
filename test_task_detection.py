import pandas as pd

from src.data_profile import load_dataset
from src.task_detection import detect_task_type


# Titanic: sınıflandırma
df = load_dataset("data/raw/titanic.csv")

task = detect_task_type(df, "Survived")
print("Titanic problem türü:", task)


# Örnek regresyon veri seti
regression_df = pd.DataFrame({
    "area": [80, 100, 120, 150, 200],
    "price": [150000, 190000, 230000, 280000, 360000]
})

task = detect_task_type(regression_df, "price")
print("Ev fiyatı problem türü:", task)