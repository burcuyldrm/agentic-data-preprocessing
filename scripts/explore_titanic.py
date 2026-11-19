from src.data_profile import (
    load_dataset,
    generate_data_profile
)

# Titanic veri setini yükle
df = load_dataset("data/raw/titanic.csv")

# Veri setinin genel bilgileri
print("Veri seti başarıyla yüklendi!")
print("Satır sayısı:", df.shape[0])
print("Sütun sayısı:", df.shape[1])

# Otomatik EDA
profile = generate_data_profile(
    df,
    target="Survived"
)

print("\n--- SÜTUN TÜRLERİ ---")
print(profile["column_types"])

print("\n--- EKSİK DEĞERLER ---")
for column, info in profile["missing_values"].items():
    if info["missing_count"] > 0:
        print(
            f"{column}: {info['missing_count']} eksik "
            f"(%{info['missing_percent']:.2f})"
        )

print("\n--- DAĞILIM ANALİZİ ---")
for column, info in profile["distribution"].items():
    print(
        f"{column}: "
        f"Ortalama={info['mean']:.2f}, "
        f"Çarpıklık={info['skewness']:.2f}"
    )

print("\n--- AYKIRI DEĞERLER ---")
for item in profile["outliers"]:
    print(
        f"{item['column']}: "
        f"{item['outlier_count']} aykırı değer"
    )