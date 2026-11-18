# Veritabanı Doğrulama Raporu

**Şema:** `docs/database/schema.sql`  
**Test tarihi:** 2026-11-05  
**SQLite sürümü:** 3.43.x  
**Test veritabanı:** `/tmp/test_agentic_preprocessing.db` (geçici, proje verilerine dokunulmadı)  
**Toplam test:** 30 | **Başarılı:** 30 | **Başarısız:** 0

---

## Test Sonuçları

| # | Test | Durum | Detay |
|---|---|---|---|
| T01 | SQL dosyası hatasız çalışıyor | PASS | `schema.sql` syntax hatası yok |
| T02 | Tüm tablolar oluşturuldu | PASS | 8 tablo: users, datasets, dataset_profiles, analysis_runs, preprocessing_strategies, evaluation_attempts, experience_memory, analysis_reports |
| T03 | `v_strategy_performance` view'u oluşturuldu | PASS | View başarıyla oluşturuldu |
| T04 | İndeksler oluşturuldu | PASS | 16 indeks |
| T05 | Örnek kullanıcı kaydı eklendi | PASS | Geçerli INSERT çalışıyor |
| T06 | Duplicate email reddedildi (UNIQUE) | PASS | `users.email` UNIQUE kısıtı aktif |
| T07 | Örnek veri seti kaydı eklendi | PASS | Geçerli INSERT çalışıyor |
| T08 | Geçersiz `download_status` reddedildi | PASS | CHECK kısıtı (`'invalid_status'` değeri reddedildi) |
| T09 | Geçersiz `source` değeri reddedildi | PASS | CHECK kısıtı (`'ftp'` değeri reddedildi) |
| T10 | Veri seti profili eklendi | PASS | JSON alanları dahil geçerli INSERT çalışıyor |
| T11 | Duplicate `dataset_profile` reddedildi | PASS | `dataset_id` UNIQUE kısıtı aktif |
| T12 | Örnek analiz çalıştırması eklendi | PASS | Geçerli INSERT çalışıyor |
| T13 | Geçersiz `experiment_mode` reddedildi | PASS | CHECK kısıtı (`'E'` değeri reddedildi) |
| T14 | Ön işleme stratejisi eklendi | PASS | Yöntem A karşılığı strateji başarıyla eklendi |
| T15 | Duplicate `strategy_signature` reddedildi | PASS | UNIQUE kısıtı aktif; eşdeğer strateji tekrar eklenemiyor |
| T16 | Değerlendirme denemesi eklendi (deneme 1) | PASS | JSON cv_scores_detail dahil INSERT çalışıyor |
| T17 | İkinci deneme farklı stratejiyle eklendi | PASS | Farklı strateji aynı analizde denenebiliyor |
| T18 | Tekrarlanan `attempt_number` reddedildi | PASS | `UNIQUE(analysis_run_id, attempt_number)` aktif |
| T19 | Tekrarlanan `strategy_id` reddedildi | PASS | `UNIQUE(analysis_run_id, strategy_id)` aktif; aynı analiz içinde aynı strateji iki kez denenemez |
| T20 | Başarısız deneme kaydedildi | PASS | `status='failed'`, `error_message` dolu; NULL CV skoru ile kayıt başarılı |
| T21 | Geçersiz `analysis_run_id` FK reddedildi | PASS | PRAGMA foreign_keys=ON ile FK kısıtı aktif |
| T22 | Deneyim belleği kaydı eklendi | PASS | Geçerli INSERT çalışıyor |
| T23 | Duplicate bellek kombinasyonu reddedildi | PASS | `UNIQUE(profile_hash, strategy_id, problem_type)` aktif |
| T24 | Nihai analiz raporu oluşturuldu | PASS | Test seti metrikleri ve CV skoru ayrı ayrı saklandı |
| T25 | Duplicate rapor reddedildi | PASS | `analysis_run_id` UNIQUE kısıtı aktif |
| T26 | Denemeler analizle doğru ilişkilendirildi | PASS | 3 deneme (2 başarılı, 1 başarısız) doğru sıralanıyor |
| T27 | Deneyim belleği sorgusu çalıştı | PASS | `v_strategy_performance` üzerinden profile_hash filtreli sorguda 2 strateji döndürüldü; KNN+robust en yüksek CV skoru (0.853, success_rate=1.0) |
| T28 | CV skoru ile test skoru birbirinden ayrı | PASS | `cv_primary_score=0.853` ≠ `final_test_primary_score=0.841`; ayrı sütunlarda saklanıyor |
| T29 | NOT NULL kısıtı (`password_hash`) çalışıyor | PASS | `password_hash` olmadan INSERT reddedildi |
| T30 | `total_missing_pct > 100` reddedildi | PASS | CHECK kısıtı (`0.0 BETWEEN 100.0`) aktif; 150.0 değeri reddedildi |

---

## Önemli Doğrulama Bulguları

### Tekrar Engeli

T18 ve T19, aynı analiz çalıştırması içinde aynı deneme numarasının ve aynı stratejinin iki kez denenemeyeceğini doğrulamaktadır. Bu iki ayrı UNIQUE kısıtı birlikte çalışarak orkestratörün tekrar engelini veritabanı düzeyinde de güvence altına alır.

### Veri Sızıntısı Önlemi

T28, `cv_primary_score` ile `final_test_primary_score` alanlarının farklı değerlere sahip olduğunu ve farklı sütunlarda saklandığını doğrulamaktadır. `final_test_primary_score` yalnızca `analysis_reports` tablosunda bulunur; `evaluation_attempts` tablosunda yer almaz. Bu tasarım, test seti sonuçlarının strateji seçimine dahil edilmesini mimari düzeyde engeller.

### Deneyim Belleği Sorgusu

T27, `v_strategy_performance` view'unun `profile_hash` filtresiyle doğru çalıştığını ve stratejileri CV skoru sırasına göre sıraladığını doğrulamaktadır. Üretim ortamında Python katmanı bu sorguyu genişleterek `last_dataset_id != ?` filtresiyle mevcut veri setinin kayıtlarını dışlayacaktır.

### Foreign Key Kısıtı

T21, `PRAGMA foreign_keys = ON` ile FK kısıtlarının aktif olduğunu doğrulamaktadır. SQLite'da bu PRAGMA bağlantı başına ayarlanmalı; uygulama kodu her bağlantı açıldığında bunu çalıştırmalıdır.

---

## Test Ortamı

```
Platform   : macOS (darwin)
Python     : 3.x
SQLite     : Python sqlite3 modülü
Veritabanı : /tmp/test_agentic_preprocessing.db (geçici)
Test verisi: Yalnızca sentetik örnek veriler
```

Proje veritabanına (`agentic_preprocessing.db`) veya gerçek kullanıcı/analiz verilerine dokunulmamıştır.
