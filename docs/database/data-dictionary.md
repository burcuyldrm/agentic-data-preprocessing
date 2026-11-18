# Veri Sözlüğü

**Proje:** Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip Çok Ajanlı Otonom Veri Analizi ve Ön İşleme Sistemi  
**Veritabanı:** SQLite (agentic_preprocessing.db)  
**Şema versiyonu:** 1.0  
**Güncelleme:** 2026-11-05

---

## İçindekiler

1. [users](#1-users)
2. [datasets](#2-datasets)
3. [dataset_profiles](#3-dataset_profiles)
4. [analysis_runs](#4-analysis_runs)
5. [preprocessing_strategies](#5-preprocessing_strategies)
6. [evaluation_attempts](#6-evaluation_attempts)
7. [experience_memory](#7-experience_memory)
8. [analysis_reports](#8-analysis_reports)
9. [v_strategy_performance (View)](#9-v_strategy_performance)

---

## 1. users

**Amaç:** Kimlik doğrulaması yapan kullanıcı hesaplarını saklar. Her kullanıcı birden fazla analiz başlatabilir.

**İlişkiler:**
- `analysis_runs.user_id` → `users.id` (1:N)

**İndeksler:** `idx_users_email` (email)

**Kullanım senaryosu:** Kullanıcı girişi, analiz sahipliği.

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz kullanıcı tanımlayıcısı |
| `email` | TEXT | — | NOT NULL | — | Kullanıcı e-posta adresi; giriş için kullanılır |
| `password_hash` | TEXT | — | NOT NULL | — | bcrypt veya argon2 ile türetilmiş parola hash'i; düz metin saklanmaz |
| `created_at` | TEXT | — | NOT NULL | `datetime('now')` | Hesap oluşturulma zamanı (UTC ISO8601) |

---

## 2. datasets

**Amaç:** OpenML, UCI veya kullanıcı yüklemesi yoluyla sisteme alınan veri setlerinin meta veri kaydı. Veri içerikleri dosya sisteminde tutulur; veritabanında yalnızca referans bilgisi saklanır.

**İlişkiler:**
- `dataset_profiles.dataset_id` → `datasets.id` (1:1)
- `analysis_runs.dataset_id` → `datasets.id` (1:N)
- `experience_memory.last_dataset_id` → `datasets.id` (opsiyonel)

**İndeksler:**
- `idx_datasets_source` (source, source_dataset_id)
- `idx_datasets_status` (download_status)
- `idx_datasets_hash` (file_hash)

**Kısıtlar:**
- `file_hash` UNIQUE: Aynı veri setinin farklı adlarla tekrar yüklenmesini önler.
- `file_path` UNIQUE: Çakışan dosya yolu oluşmaz.

**Kullanım senaryosu:** Research Agent tarafından bulunan ve indirilen veri setlerinin kaydı; deduplication kontrolü.

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz veri seti tanımlayıcısı |
| `source` | TEXT | — | NOT NULL | — | Kaynak: `openml`, `uci`, `upload` |
| `source_dataset_id` | TEXT | — | NULL | — | Kaynak sistemdeki tanımlayıcı (örn. OpenML dataset ID); yüklemeler için NULL |
| `name` | TEXT | — | NOT NULL | — | Veri seti adı (örn. "adult", "iris") |
| `file_path` | TEXT | — | NOT NULL | — | Yerel dosya sistemi yolu (örn. `data/datasets/adult.csv`) |
| `file_hash` | TEXT | — | NOT NULL | — | SHA-256 hash; tekrar yükleme ve sahte kopyaları engeller |
| `row_count` | INTEGER | — | NULL | — | Toplam satır sayısı; indirme tamamlandıktan sonra doldurulur |
| `column_count` | INTEGER | — | NULL | — | Toplam sütun sayısı |
| `file_size_bytes` | INTEGER | — | NULL | — | Dosya boyutu (bayt) |
| `download_status` | TEXT | — | NOT NULL | `'pending'` | İndirme durumu; izin verilen değerler aşağıda |
| `created_at` | TEXT | — | NOT NULL | `datetime('now')` | Kayıt oluşturulma zamanı (UTC ISO8601) |

**`download_status` izin verilen değerler:**

| Değer | Açıklama |
|---|---|
| `pending` | İndirme henüz başlamadı |
| `downloading` | İndirme devam ediyor |
| `ready` | İndirme tamamlandı, doğrulama başarılı |
| `failed` | İndirme başarısız oldu |
| `invalid` | Dosya bozuk, desteklenmeyen format veya boyut sınırı aşıldı |

---

## 3. dataset_profiles

**Amaç:** Veri setinin kalıcı profil bilgisi. Profil, ilk analizde eğitim verisi üzerinde hesaplanır. `profile_hash` alanı, benzer veri setlerini bulmak için deneyim belleğiyle eşleştirilir.

**İlişkiler:**
- `dataset_profiles.dataset_id` → `datasets.id` (1:1, UNIQUE)

**İndeksler:**
- `idx_profiles_hash` (profile_hash)
- `idx_profiles_problem_type` (problem_type)

**Kısıtlar:**
- `dataset_id` UNIQUE: Her veri setinin en fazla bir profili olabilir.
- `total_missing_pct` CHECK: 0.0 ile 100.0 arasında olmalıdır.
- `class_count` CHECK: Yalnızca sınıflandırma problemlerinde, en az 2 sınıf.

**Tasarım kararı:** Profil veri setine bağlıdır, analiz çalıştırmasına değil. Profil eğitim verisi üzerinde hesaplanmasına karşın, aynı veri seti farklı analizlerde kullanıldığında tutarlı bir profil referansı gereklidir. Farklı rastgelelik tohumlarından kaynaklanan küçük farklılıklar profil hash'ini etkilemez; profile_hash temel istatistiklerden türetilir.

**JSON alanları:**

`missing_value_detail` örneği:
```json
{"age": 5.2, "occupation": 0.0, "native_country": 1.8}
```

`class_balance_detail` örneği (sınıflandırma):
```json
{"<=50K": 24720, ">50K": 7841}
```

`numeric_stats` örneği:
```json
{
  "age": {"mean": 38.4, "std": 13.1, "min": 17.0, "max": 90.0, "median": 37.0},
  "hours_per_week": {"mean": 40.4, "std": 12.3, "min": 1.0, "max": 99.0, "median": 40.0}
}
```

`categorical_stats` örneği:
```json
{
  "occupation": {"unique_count": 14, "rare_category_pct": 2.3},
  "education": {"unique_count": 16, "rare_category_pct": 0.5}
}
```

`skewness_detail` örneği:
```json
{"capital_gain": 11.9, "capital_loss": 4.6, "age": 0.6}
```

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz profil tanımlayıcısı |
| `dataset_id` | INTEGER | FK | NOT NULL | — | Profili ait olduğu veri seti (`datasets.id`); UNIQUE |
| `numeric_column_count` | INTEGER | — | NOT NULL | — | Sayısal sütun adedi |
| `categorical_column_count` | INTEGER | — | NOT NULL | — | Kategorik sütun adedi |
| `total_missing_pct` | REAL | — | NOT NULL | — | Tüm hücrelerdeki ortalama eksik değer yüzdesi |
| `missing_value_detail` | TEXT | — | NULL | — | JSON: sütun bazında eksik değer yüzdeleri |
| `target_column` | TEXT | — | NOT NULL | — | Hedef değişken sütun adı |
| `problem_type` | TEXT | — | NOT NULL | — | Problem türü: `classification` veya `regression` |
| `class_count` | INTEGER | — | NULL | — | Sınıf sayısı; yalnızca sınıflandırma için |
| `class_balance_detail` | TEXT | — | NULL | — | JSON: sınıf dağılımı; yalnızca sınıflandırma için |
| `train_row_count` | INTEGER | — | NULL | — | Eğitim kümesindeki satır sayısı (%80) |
| `test_row_count` | INTEGER | — | NULL | — | Test kümesindeki satır sayısı (%20) |
| `numeric_stats` | TEXT | — | NULL | — | JSON: sayısal sütunların özet istatistikleri |
| `categorical_stats` | TEXT | — | NULL | — | JSON: kategorik sütunların özet istatistikleri |
| `skewness_detail` | TEXT | — | NULL | — | JSON: sayısal sütunların çarpıklık değerleri |
| `profile_hash` | TEXT | — | NOT NULL | — | Profil özelliklerinden türetilen hash; benzerlik aramasında kullanılır |
| `profiled_at` | TEXT | — | NOT NULL | `datetime('now')` | Profil oluşturulma zamanı (UTC ISO8601) |

---

## 4. analysis_runs

**Amaç:** Kullanıcının başlattığı her analiz çalıştırmasının kaydı. A/B/C/D deney modunu, maksimum deneme sayısını, rastgelelik tohumunu ve analiz durumunu içerir.

**İlişkiler:**
- `analysis_runs.user_id` → `users.id` (N:1)
- `analysis_runs.dataset_id` → `datasets.id` (N:1)
- `evaluation_attempts.analysis_run_id` → `analysis_runs.id` (1:N)
- `analysis_reports.analysis_run_id` → `analysis_runs.id` (1:1)
- `experience_memory.last_analysis_run_id` → `analysis_runs.id` (opsiyonel)

**İndeksler:**
- `idx_runs_user` (user_id)
- `idx_runs_dataset` (dataset_id)
- `idx_runs_status` (status)

**Kısıtlar:**
- `attempt_count <= max_attempts` CHECK: Deneme sayısı sınırı aşılamaz.
- `max_attempts` CHECK: 1 ile 10 arasında.

**Kullanım senaryosu:** Orkestratör her yeni analiz için bu tabloya satır ekler; durum alanı iş akışı boyunca güncellenir.

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz analiz çalıştırma tanımlayıcısı |
| `user_id` | INTEGER | FK | NOT NULL | — | Analizi başlatan kullanıcı (`users.id`) |
| `dataset_id` | INTEGER | FK | NOT NULL | — | Kullanılan veri seti (`datasets.id`) |
| `user_request` | TEXT | — | NOT NULL | — | Kullanıcının serbest metin analiz isteği |
| `status` | TEXT | — | NOT NULL | `'pending'` | Analiz durumu; izin verilen değerler aşağıda |
| `experiment_mode` | TEXT | — | NOT NULL | `'D'` | Deney modu: A, B, C veya D |
| `random_seed` | INTEGER | — | NOT NULL | `42` | Eğitim/test bölme rastgelelik tohumu |
| `max_attempts` | INTEGER | — | NOT NULL | `5` | Maksimum strateji deneme sayısı (1–10) |
| `attempt_count` | INTEGER | — | NOT NULL | `0` | Tamamlanan deneme sayısı |
| `started_at` | TEXT | — | NOT NULL | `datetime('now')` | Analiz başlangıç zamanı (UTC ISO8601) |
| `finished_at` | TEXT | — | NULL | — | Analiz bitiş zamanı; tamamlanmamışsa NULL |
| `error_message` | TEXT | — | NULL | — | Analiz başarısız olduysa hata mesajı |

**`status` izin verilen değerler:**

| Değer | Açıklama |
|---|---|
| `pending` | Analiz kuyruğa alındı |
| `running` | Analiz devam ediyor |
| `completed` | Tüm denemeler tamamlandı, rapor oluşturuldu |
| `failed` | Kurtarılamaz hata ile sonlandı |
| `cancelled` | Kullanıcı tarafından iptal edildi |

**`experiment_mode` izin verilen değerler:**

| Değer | Açıklama |
|---|---|
| `A` | Sabit ön işleme (ortanca doldurma + one-hot + standart ölçekleme) |
| `B` | Tek aşamalı LLM strateji seçimi, yeniden deneme yok |
| `C` | Geri bildirimli seçim, en fazla 5 farklı strateji |
| `D` | Deneyim belleği destekli seçim, en fazla 5 farklı strateji |

---

## 5. preprocessing_strategies

**Amaç:** Denenmiş veya sisteme tanımlı ön işleme stratejilerinin global kataloğu. `strategy_signature` alanı sayesinde aynı kombinasyon birden fazla kez eklenmez. Bu tablo global paylaşımlıdır; bir stratejinin hangi analizde denendiği `evaluation_attempts` üzerinden izlenir.

**İlişkiler:**
- `evaluation_attempts.strategy_id` → `preprocessing_strategies.id` (1:N)
- `experience_memory.strategy_id` → `preprocessing_strategies.id` (1:N)
- `analysis_reports.best_strategy_id` → `preprocessing_strategies.id` (opsiyonel)

**İndeksler:**
- `idx_strategies_signature` (strategy_signature)

**Kısıtlar:**
- `strategy_signature` UNIQUE: Eşdeğer stratejilerin tekrar eklenmesini önler.

**Tekrar engeli tasarımı:** `evaluation_attempts` tablosundaki `UNIQUE (analysis_run_id, strategy_id)` kısıtı, aynı analiz içinde aynı stratejinin iki kez denenmesini veritabanı düzeyinde engeller. `strategy_signature`, farklı yollarla oluşturulan ancak özdeş parametre setlerine sahip stratejilerin tek bir satır olarak temsil edilmesini sağlar.

**JSON alanı:**

`extra_params` örneği:
```json
{"knn_neighbors": 5, "selectkbest_k": 10, "variance_threshold": 0.01}
```

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz strateji tanımlayıcısı |
| `imputation_numeric` | TEXT | — | NOT NULL | — | Sayısal eksik değer yöntemi; izin verilen değerler aşağıda |
| `imputation_categorical` | TEXT | — | NOT NULL | — | Kategorik eksik değer yöntemi; izin verilen değerler aşağıda |
| `scaling_method` | TEXT | — | NOT NULL | — | Ölçekleme yöntemi; izin verilen değerler aşağıda |
| `encoding_method` | TEXT | — | NOT NULL | — | Kodlama yöntemi; izin verilen değerler aşağıda |
| `outlier_handling` | TEXT | — | NOT NULL | `'none'` | Aykırı değer yönetimi; izin verilen değerler aşağıda |
| `feature_selection` | TEXT | — | NOT NULL | `'none'` | Özellik seçimi; izin verilen değerler aşağıda |
| `extra_params` | TEXT | — | NULL | — | JSON: ek parametreler (örn. KNN komşu sayısı) |
| `strategy_signature` | TEXT | — | NOT NULL | — | Parametrelerin deterministik hash'i; UNIQUE |
| `created_at` | TEXT | — | NOT NULL | `datetime('now')` | Strateji ilk tanımlanma zamanı (UTC ISO8601) |

**İzin verilen değerler:**

| Sütun | Değerler |
|---|---|
| `imputation_numeric` | `mean`, `median`, `knn`, `iterative`, `constant` |
| `imputation_categorical` | `most_frequent`, `constant`, `knn` |
| `scaling_method` | `standard`, `robust`, `minmax`, `none` |
| `encoding_method` | `onehot`, `ordinal`, `target`, `none` |
| `outlier_handling` | `clip`, `remove`, `none` |
| `feature_selection` | `none`, `variance_threshold`, `selectkbest` |

---

## 6. evaluation_attempts

**Amaç:** Bir analiz çalıştırması içindeki her değerlendirme denemesinin ayrıntılı kaydı. 5 katlı çapraz doğrulama skoru, süre, LLM çağrı sayısı, durum ve hata bilgisi saklanır.

**İlişkiler:**
- `evaluation_attempts.analysis_run_id` → `analysis_runs.id` (N:1)
- `evaluation_attempts.strategy_id` → `preprocessing_strategies.id` (N:1)

**İndeksler:**
- `idx_attempts_run` (analysis_run_id)
- `idx_attempts_strategy` (strategy_id)
- `idx_attempts_status` (status)

**Kısıtlar:**
- `UNIQUE (analysis_run_id, attempt_number)`: Bir analizde deneme numarası tekrar edilemez.
- `UNIQUE (analysis_run_id, strategy_id)`: Bir analizde aynı strateji iki kez denenemez. Tekrar algılandığında orkestratör bu kısıta teslim olmadan Python katmanında engeller; kısıt son güvence olarak işlev görür.

**JSON alanı:**

`cv_scores_detail` örneği:
```json
{
  "fold_scores": [0.823, 0.847, 0.791, 0.882, 0.831],
  "mean": 0.835,
  "std": 0.031,
  "metric": "macro_f1"
}
```

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz deneme tanımlayıcısı |
| `analysis_run_id` | INTEGER | FK | NOT NULL | — | Ait olduğu analiz (`analysis_runs.id`) |
| `strategy_id` | INTEGER | FK | NOT NULL | — | Denenen strateji (`preprocessing_strategies.id`) |
| `attempt_number` | INTEGER | — | NOT NULL | — | Analiz içindeki sıra numarası (1'den başlar) |
| `model_type` | TEXT | — | NOT NULL | — | Kullanılan model; izin verilen değerler aşağıda |
| `primary_metric_name` | TEXT | — | NOT NULL | — | Birincil değerlendirme metriği adı; izin verilen değerler aşağıda |
| `cv_primary_score` | REAL | — | NULL | — | 5 katlı CV birincil metrik ortalaması; hata durumunda NULL |
| `cv_secondary_score` | REAL | — | NULL | — | 5 katlı CV ikincil metrik ortalaması; hata durumunda NULL |
| `cv_scores_detail` | TEXT | — | NULL | — | JSON: kat bazında skolar, ortalama, standart sapma |
| `duration_seconds` | REAL | — | NULL | — | Deneme toplam süresi (saniye) |
| `llm_call_count` | INTEGER | — | NOT NULL | `0` | Bu deneme için yapılan LLM çağrısı sayısı |
| `status` | TEXT | — | NOT NULL | `'running'` | Deneme durumu; izin verilen değerler aşağıda |
| `error_message` | TEXT | — | NULL | — | Hata durumunda mesaj; başarılı denemelerde NULL |
| `attempted_at` | TEXT | — | NOT NULL | `datetime('now')` | Deneme başlangıç zamanı (UTC ISO8601) |

**İzin verilen değerler:**

| Sütun | Değerler |
|---|---|
| `model_type` | `logistic_regression` (sınıflandırma), `ridge_regression` (regresyon) |
| `primary_metric_name` | `macro_f1`, `balanced_accuracy` (sınıflandırma); `rmse`, `mae` (regresyon) |
| `status` | `running`, `success`, `failed`, `invalid`, `skipped` |

---

## 7. experience_memory

**Amaç:** Projenin temel araştırma bileşeni. Farklı veri setlerinde gerçekleştirilen stratejilerin toplulaştırılmış performansını saklar. `profile_hash` üzerinden benzer veri setleri için strateji önerileri üretilir. Her `(profile_hash, strategy_id, problem_type)` kombinasyonu için tek bir satır tutulur; her yeni deneme sonucunda Python katmanı bu satırı UPSERT ile günceller.

**İlişkiler:**
- `experience_memory.strategy_id` → `preprocessing_strategies.id` (N:1)
- `experience_memory.last_dataset_id` → `datasets.id` (opsiyonel, denetim)
- `experience_memory.last_analysis_run_id` → `analysis_runs.id` (opsiyonel, denetim)

**İndeksler:**
- `idx_memory_profile` (profile_hash, problem_type) — benzerlik aramasında kullanılır
- `idx_memory_score` (cv_score_avg DESC) — en başarılı stratejilere hızlı erişim

**Kısıtlar:**
- `UNIQUE (profile_hash, strategy_id, problem_type)`: Bir kombinasyon için tek satır.
- `success_count <= attempt_count` CHECK: Başarı sayısı toplam deneme sayısını geçemez.

**Veri sızıntısı önlemi:** Bellek sorgulanırken Python katmanı, geçerli veri setine ait kayıtları `last_dataset_id` veya `datasets.file_hash` üzerinden dışlar. Bu kural, bir veri setinin kendi önceki performansından faydalanmasını engeller.

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz bellek kaydı tanımlayıcısı |
| `profile_hash` | TEXT | — | NOT NULL | — | `dataset_profiles.profile_hash` ile örtüşür; benzerlik araması için kullanılır |
| `strategy_id` | INTEGER | FK | NOT NULL | — | İlgili strateji (`preprocessing_strategies.id`) |
| `problem_type` | TEXT | — | NOT NULL | — | Problem türü: `classification` veya `regression` |
| `cv_score_avg` | REAL | — | NOT NULL | — | Tüm denemeler üzerinden birincil CV metriği ortalaması |
| `cv_score_best` | REAL | — | NOT NULL | — | Bu kombinasyon için elde edilen en yüksek CV skoru |
| `attempt_count` | INTEGER | — | NOT NULL | `1` | Bu kombinasyonun toplam deneme sayısı |
| `success_count` | INTEGER | — | NOT NULL | `0` | Başarılı (referansı geçen) deneme sayısı |
| `avg_duration_seconds` | REAL | — | NULL | — | Ortalama deneme süresi (saniye) |
| `last_dataset_id` | INTEGER | FK | NULL | — | Son katkıda bulunan veri seti (denetim) |
| `last_analysis_run_id` | INTEGER | FK | NULL | — | Son katkıda bulunan analiz çalıştırması (denetim) |
| `last_updated_at` | TEXT | — | NOT NULL | `datetime('now')` | Son güncelleme zamanı (UTC ISO8601) |

---

## 8. analysis_reports

**Amaç:** Bir analiz çalıştırmasının nihai raporu. Çapraz doğrulama skoru ile test seti sonuçları birbirinden açıkça ayrılır. Test seti sonuçları strateji seçiminden sonra bir kez hesaplanır ve yalnızca burada saklanır.

**İlişkiler:**
- `analysis_reports.analysis_run_id` → `analysis_runs.id` (1:1, UNIQUE)
- `analysis_reports.best_strategy_id` → `preprocessing_strategies.id` (opsiyonel)

**İndeksler:** `idx_reports_run` (analysis_run_id)

**Kısıtlar:**
- `analysis_run_id` UNIQUE: Her analiz çalıştırmasının en fazla bir raporu olabilir.

**Test seti güvenliği:** `final_test_primary_score` ve `final_test_secondary_score` alanları, en iyi strateji seçildikten sonra test seti üzerinde **bir kez** hesaplanır. Bu değerler `evaluation_attempts` tablosunda yer almaz; böylece test sonuçlarının strateji seçimine dahil edilmesi mimaride önlenmiş olur.

**JSON alanı:**

`final_test_metrics_detail` örneği:
```json
{
  "macro_f1": 0.841,
  "balanced_accuracy": 0.859,
  "confusion_matrix": [[4823, 312], [589, 1476]]
}
```

| Sütun | SQLite Türü | PK/FK | NULL | Varsayılan | Açıklama |
|---|---|---|---|---|---|
| `id` | INTEGER | PK | NOT NULL | AUTOINCREMENT | Benzersiz rapor tanımlayıcısı |
| `analysis_run_id` | INTEGER | FK | NOT NULL | — | İlgili analiz (`analysis_runs.id`); UNIQUE |
| `best_strategy_id` | INTEGER | FK | NULL | — | Seçilen en iyi strateji; geçerli strateji yoksa NULL |
| `final_test_primary_score` | REAL | — | NULL | — | Test seti birincil metrik skoru (strateji seçiminden sonra hesaplanır) |
| `final_test_secondary_score` | REAL | — | NULL | — | Test seti ikincil metrik skoru |
| `final_test_metrics_detail` | TEXT | — | NULL | — | JSON: test seti tüm metrikleri |
| `cv_primary_score` | REAL | — | NULL | — | En iyi denemenin CV skoru (referans karşılaştırması için) |
| `total_attempts` | INTEGER | — | NOT NULL | `0` | Toplam deneme sayısı |
| `total_duration_seconds` | REAL | — | NULL | — | Analiz toplam süresi (saniye) |
| `total_llm_calls` | INTEGER | — | NOT NULL | `0` | Analiz boyunca yapılan toplam LLM çağrısı sayısı |
| `selection_rationale` | TEXT | — | NULL | — | LLM tarafından üretilen strateji seçim gerekçesi |
| `report_created_at` | TEXT | — | NOT NULL | `datetime('now')` | Rapor oluşturulma zamanı (UTC ISO8601) |

---

## 9. v_strategy_performance

**Tür:** VIEW (salt okunur)

**Amaç:** `experience_memory` ile `preprocessing_strategies` tablolarını birleştirir. Benzer profil için strateji sıralaması yapılırken kullanılır. `success_rate` hesaplaması bu view içinde yapılır.

**Kullanım senaryosu:** Yeni bir analiz başlarken, geçerli veri setinin `profile_hash` değerine benzer profil_hash'ler için en başarılı stratejilerin listelenmesi.

| Sütun | Kaynak | Açıklama |
|---|---|---|
| `memory_id` | experience_memory.id | Bellek kaydı tanımlayıcısı |
| `profile_hash` | experience_memory.profile_hash | Profil hash değeri |
| `problem_type` | experience_memory.problem_type | Problem türü |
| `strategy_id` | preprocessing_strategies.id | Strateji tanımlayıcısı |
| `strategy_signature` | preprocessing_strategies.strategy_signature | Strateji hash'i |
| `imputation_numeric` | preprocessing_strategies.imputation_numeric | Sayısal impütasyon yöntemi |
| `imputation_categorical` | preprocessing_strategies.imputation_categorical | Kategorik impütasyon yöntemi |
| `scaling_method` | preprocessing_strategies.scaling_method | Ölçekleme yöntemi |
| `encoding_method` | preprocessing_strategies.encoding_method | Kodlama yöntemi |
| `outlier_handling` | preprocessing_strategies.outlier_handling | Aykırı değer yönetimi |
| `feature_selection` | preprocessing_strategies.feature_selection | Özellik seçimi |
| `cv_score_avg` | experience_memory.cv_score_avg | Ortalama CV skoru |
| `cv_score_best` | experience_memory.cv_score_best | En yüksek CV skoru |
| `attempt_count` | experience_memory.attempt_count | Toplam deneme sayısı |
| `success_count` | experience_memory.success_count | Başarılı deneme sayısı |
| `success_rate` | hesaplanmış | success_count / attempt_count (0.0–1.0) |
| `avg_duration_seconds` | experience_memory.avg_duration_seconds | Ortalama deneme süresi |
| `last_updated_at` | experience_memory.last_updated_at | Son güncelleme zamanı |
