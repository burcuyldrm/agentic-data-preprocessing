# Veritabanı Tasarımı

**Proje:** Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip Çok Ajanlı Otonom Veri Analizi ve Ön İşleme Sistemi  
**Geliştirici:** Burcu Yıldırım — Yapay Zeka ve Veri Mühendisliği Tasarımı Dersi

---

## 1. Genel Amaç

Bu veritabanı, çok ajanlı ön işleme sisteminin kalıcı durumunu saklar. Temel işlevler:

- Kullanıcı hesapları ve kimlik doğrulama bilgilerini yönetmek
- Araştırılan ve indirilen veri setlerinin meta verilerini kaydetmek
- Her analiz çalıştırmasını ve deneme geçmişini takip etmek
- Deneyim belleğini saklamak: geçmiş veri setlerinde hangi stratejilerin ne kadar başarılı olduğunu

---

## 2. Temel Varlıklar ve İlişkiler

```
users (1) ──────────── (N) analysis_runs
datasets (1) ─────────── (1) dataset_profiles
datasets (1) ─────────── (N) analysis_runs
analysis_runs (1) ────── (N) evaluation_attempts
preprocessing_strategies (1) ── (N) evaluation_attempts
preprocessing_strategies (1) ── (N) experience_memory
analysis_runs (1) ────── (0..1) analysis_reports
```

| Tablo | Açıklama |
|---|---|
| `users` | Kimlik doğrulaması yapan kullanıcı hesapları |
| `datasets` | Veri seti meta verisi; içerik dosya sisteminde |
| `dataset_profiles` | Veri seti özellik profili; benzerlik aramasında kullanılır |
| `analysis_runs` | Her analiz isteğinin kaydı; deney modu ve deneme sınırını içerir |
| `preprocessing_strategies` | Global strateji kataloğu; tekrarlayan stratejileri önler |
| `evaluation_attempts` | Her değerlendirme denemesinin ayrıntılı kaydı |
| `experience_memory` | Farklı veri setlerindeki toplulaştırılmış strateji performansı |
| `analysis_reports` | Nihai rapor; CV ve test seti sonuçları ayrı ayrı saklanır |

---

## 3. Neden SQLite?

Proje öneri formunda açıkça belirtilmiştir. Teknik gerekçeler:

- Sıfır ek konfigürasyon; tek dosya (`agentic_preprocessing.db`)
- Tek geliştirici ve 10'dan az eş zamanlı bağlantı için yeterli kapasite
- WAL modu ile okuma/yazma eşzamanlılığı
- Üretim ortamına geçişte PostgreSQL ile uyumlu SQL sözdizimi

---

## 4. Deneyim Belleğinin Temsili

`experience_memory` tablosu, projenin temel araştırma bileşenidir. Her `(profile_hash, strategy_id, problem_type)` kombinasyonu için tek bir satır tutulur.

**Temel mekanizma:**
1. Yeni bir analiz başlarken, geçerli veri setinin `profile_hash` değeri `dataset_profiles` tablosundan alınır.
2. `v_strategy_performance` view'u üzerinden aynı veya benzer profile_hash'e sahip geçmiş denemeler sorgulanır.
3. Python katmanı bu sorguya `last_dataset_id != <geçerli_dataset_id>` filtresini ekleyerek mevcut veri setinin kendi tarihçesinden faydalanmasını önler.
4. Her deneme sonucunda Python katmanı, `experience_memory` tablosundaki ilgili satırı UPSERT ile günceller.

**Benzerlik araması:** `profile_hash` tam eşleşme için kullanılır. Farklı hash'li ama benzer profillere sahip veri setleri için benzerlik skoru Python katmanında hesaplanır; SQL yalnızca ön filtreleme yapar.

---

## 5. Değerlendirme Denemelerinin Saklanması

`evaluation_attempts` tablosunun iki temel UNIQUE kısıtı vardır:

- `UNIQUE (analysis_run_id, attempt_number)`: Bir analiz içinde deneme numarası tekrar edilemez.
- `UNIQUE (analysis_run_id, strategy_id)`: Bir analiz içinde aynı strateji iki kez denenemez.

Başarısız denemeler `status='failed'`, `error_message` dolu ve NULL CV skoru ile kaydedilir. Bu kayıtlar deneyim belleğine de aktarılır; böylece gelecekte aynı strateji benzer veri setlerinde denenmeden önce geçmişteki başarısızlıklar görülebilir.

---

## 6. Veri Seti Profillerinin İlişkilendirilmesi

`dataset_profiles` tablosu `datasets.id` üzerinden bağlıdır (1:1, UNIQUE). Profil eğitim verisi üzerinde hesaplanır ve veri setine ait kalıcı bir özellik olarak saklanır. Farklı analizlerde aynı veri seti için farklı random_seed kullanılabilir; bu durum profil hesaplamalarında küçük farklılıklara yol açabilir. Ancak `profile_hash`, temel istatistiklerden (sütun sayıları, eksik oran, problem türü) türetildiğinden rastgelelik tohumundan bağımsızdır.

---

## 7. Nihai Test Sonuçlarının Ayrılması

Test seti sonuçları yalnızca `analysis_reports` tablosunda saklanır:

| Alan | Tablo | Açıklama |
|---|---|---|
| `cv_primary_score` | `analysis_reports` | En iyi denemenin CV skoru; strateji seçiminde kullanıldı |
| `final_test_primary_score` | `analysis_reports` | Test seti birincil metriği; strateji seçiminden *sonra* hesaplandı |
| `final_test_secondary_score` | `analysis_reports` | Test seti ikincil metriği |

`evaluation_attempts` tablosunda test seti metrikleri yer almaz. Bu tasarım, test sonuçlarının değerlendirme döngüsüne dahil edilmesini mimari düzeyde önler.

---

## 8. Veri Bütünlüğü

| Mekanizma | Tablolar | Amaç |
|---|---|---|
| PRIMARY KEY | Tüm tablolar | Benzersiz satır tanımlayıcısı |
| FOREIGN KEY | analysis_runs, evaluation_attempts, vb. | Tutarsız referansları engeller |
| UNIQUE | email, file_hash, strategy_signature, vb. | Veri tekrarını önler |
| CHECK | download_status, source, problem_type, vb. | İzin verilen değer kümelerini zorlar |
| NOT NULL | Kritik alanlar | Zorunlu verilerin eksikliğini engeller |
| PRAGMA foreign_keys = ON | Bağlantı başına | SQLite'ta FK kısıtlarını etkinleştirir |

---

## 9. Şemanın Çalıştırılması

```bash
# Yeni veritabanı oluştur
sqlite3 agentic_preprocessing.db < docs/database/schema.sql

# Bağlantıda foreign_keys'i etkinleştir (her bağlantıda yapılmalı)
sqlite3 agentic_preprocessing.db "PRAGMA foreign_keys = ON;"

# Tabloları doğrula
sqlite3 agentic_preprocessing.db ".tables"
```

**Python bağlantısı:**
```python
import sqlite3

con = sqlite3.connect("agentic_preprocessing.db")
con.execute("PRAGMA foreign_keys = ON")
con.execute("PRAGMA journal_mode = WAL")
```

---

## 10. Olası Genişletmeler

Mevcut şema birinci sürümü kapsamaktadır. Makul gelecek genişletmeler:

| Genişletme | Açıklama |
|---|---|
| Veri seti etiketleme | `datasets` tablosuna `tags` JSON sütunu eklenebilir |
| Çoklu model desteği | `evaluation_attempts.model_type` CHECK kısıtına yeni modeller eklenebilir |
| Profil benzerlik önbelleği | Benzerlik skorları bir `profile_similarity_cache` tablosunda önbelleğe alınabilir |
| API anahtarı yönetimi | LLM sağlayıcı bilgileri için `llm_configs` tablosu eklenebilir |
| Kullanıcı rolleri | `users` tablosuna `role` sütunu eklenerek çok kullanıcılı yönetim sağlanabilir |

---

## Dosyalar

| Dosya | Açıklama |
|---|---|
| `er-diagram.mmd` | Mermaid ER diyagramı kaynak dosyası |
| `er-diagram.svg` | ER diyagramı görsel çıktısı |
| `schema.sql` | SQLite şeması (doğrudan çalıştırılabilir) |
| `data-dictionary.md` | Tüm tablo ve sütunların açıklamaları |
| `validation-report.md` | 30 testin sonuçları |
| `README.md` | Bu dosya — tasarım kararları |
