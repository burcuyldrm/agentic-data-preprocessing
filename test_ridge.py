import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

from src.data_split import split_dataset
from src.preprocessing import build_baseline_preprocessor


# 1. Örnek regresyon veri seti
df = pd.DataFrame({
    "area": [80, 100, 120, 150, 200, 90, 110, 140, 170, 220],
    "rooms": [2, 3, 3, 4, 5, 2, 3, 4, 4, 5],
    "price": [
        150000, 190000, 230000, 280000, 360000,
        170000, 210000, 260000, 320000, 400000
    ]
})

# 2. Eğitim/test ayrımı
X_train, X_test, y_train, y_test = split_dataset(
    df,
    target="price",
    task_type="regression"
)

# 3. Ön işleme
preprocessor = build_baseline_preprocessor(
    numeric_columns=["area", "rooms"],
    categorical_columns=[]
)

# 4. Model Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0))
])

# 5. Modeli eğit
model.fit(X_train, y_train)

# 6. Tahmin yap
y_pred = model.predict(X_test)

# 7. Performans metrikleri
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("RIDGE REGRESSION - A BASELINE")
print("-----------------------------")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")

print("\nGerçek ve tahmin edilen fiyatlar:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Gerçek: {actual}, Tahmin: {predicted:.2f}")