from src.data_profile import load_dataset, detect_column_types
from src.data_profile import (
    load_dataset,
    detect_column_types,
    analyze_missing_values,
    analyze_distribution,
    analyze_outliers,
    generate_data_profile
)

df = load_dataset("data/raw/sample.csv")

print("Veri seti başarıyla yüklendi!")
print("Satır sayısı:", df.shape[0])
print("Sütun sayısı:", df.shape[1])

column_types = detect_column_types(df, target="purchased")

print("\nSayısal sütunlar:")
print(column_types["numeric"])

print("\nKategorik sütunlar:")
print(column_types["categorical"])

print("\nEksik Değer Analizi:")
missing_report = analyze_missing_values(df)
print(missing_report)

print("\nVeri Dağılımı Analizi:")
distribution_report = analyze_distribution(df, target="purchased")
print(distribution_report)

print("\nAykırı Değer Analizi:")
outlier_report = analyze_outliers(df, target="purchased")
print(outlier_report)

print("\nOtomatik Veri Profili:")

profile = generate_data_profile(
    df,
    target="purchased"
)

print("Veri seti bilgileri:")
print(profile["dataset_info"])

print("\nSütun türleri:")
print(profile["column_types"])

print("\nEksik değerler:")
print(profile["missing_values"])