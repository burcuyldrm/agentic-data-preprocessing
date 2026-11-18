# Sistem Tasarım Diyagramları

Bu klasör, **Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip Çok Ajanlı Otonom Veri Analizi ve Ön İşleme Sistemi** projesinin sistem tasarım diyagramlarını içermektedir.

Geliştirici: Burcu Yıldırım — Yapay Zeka ve Veri Mühendisliği Tasarımı Dersi

---

## Diyagram Listesi

| # | Diyagram | Format | Kaynak | SVG |
|---|----------|--------|--------|-----|
| 1 | UML Use Case | PlantUML | [use-case.puml](01-use-case/use-case.puml) | [use-case.svg](01-use-case/use-case.svg) |
| 2 | Sistem Mimarisi | Mermaid | [architecture.mmd](02-system-architecture/architecture.mmd) | [architecture.svg](02-system-architecture/architecture.svg) |
| 3 | Veri Seti Keşfi ve Doğrulama | Mermaid | [dataset-discovery.mmd](03-dataset-discovery/dataset-discovery.mmd) | [dataset-discovery.svg](03-dataset-discovery/dataset-discovery.svg) |
| 4 | Veri Profilleme ve Ön İşleme | Mermaid | [preprocessing.mmd](04-data-preprocessing/preprocessing.mmd) | [preprocessing.svg](04-data-preprocessing/preprocessing.svg) |
| 5 | Değerlendirme, Özdenetim ve Bellek | Mermaid | [evaluation.mmd](05-evaluation-self-correction/evaluation.mmd) | [evaluation.svg](05-evaluation-self-correction/evaluation.svg) |
| 6 | Uçtan Uca Yapay Zeka Pipeline | Mermaid | [ai-pipeline.mmd](06-ai-pipeline/ai-pipeline.mmd) | [ai-pipeline.svg](06-ai-pipeline/ai-pipeline.svg) |

---

## Diyagram 1 — UML Use Case

**Araç:** PlantUML 1.2026.9 | **Dosya:** `01-use-case/`

### Amaç
Sistemin dış aktörlerle olan etkileşimlerini UML 2.5 standardına uygun biçimde gösterir. Kimlik yönetiminden sonuç raporlarına kadar kullanıcının sisteme erişebileceği tüm kullanım senaryolarını kapsar.

### Aktörler
- **Kullanıcı** (insan aktör): Sistemin tüm işlevlerine erişen birincil aktör.
- **OpenML Servisi** (dış sistem): Veri seti aramasında sorgulanan açık veri kaynağı.
- **UCI ML Repository** (dış sistem): Veri seti aramasında sorgulanan açık veri kaynağı.
- **LLM API Servisi** (dış sistem): Otonom analiz sırasında ajanlara karar desteği sağlayan büyük dil modeli servisi.

> Üç LLM destekli ajan (Research Agent, Analysis & Preprocessing Agent, Evaluation & Self-Correction Agent) **sistem sınırı içindedir** ve aktör olarak gösterilmemektedir. İç bileşenler Diyagram 2'de ayrıntılandırılmaktadır.

### Kullanım Senaryoları (15 adet)
4 paket altında gruplandırılmıştır: Kimlik Yönetimi, Veri Seti Keşfi, Analiz Yapılandırma, İzleme ve Sonuçlar.

### Include / Extend İlişkileri
- `Analiz İsteği Oluştur` **«include»** `Veri Seti Ara`: Her analiz isteği zorunlu olarak kaynak aramayı kapsar.
- `Hedef Değişken ve Problem Türü Belirle` **«extend»** `Veri Seti Seç`: Veri seti seçimini genişleten isteğe bağlı yapılandırma adımı.

### Güncelleme
PlantUML ile güncellenir: `plantuml -tsvg 01-use-case/use-case.puml`

---

## Diyagram 2 — Sistem Mimarisi

**Araç:** Mermaid 11.17.1 | **Dosya:** `02-system-architecture/`

### Amaç
Sistemin katmanlı bileşen mimarisini, bileşenler arası iletişim yönlerini ve dış servis entegrasyonlarını gösterir.

### Katmanlar (6 adet)
| Katman | Bileşenler |
|--------|-----------|
| Sunum | Web Arayüzü |
| API | FastAPI, JWT Kimlik Doğrulama, Endpoint'ler |
| Orkestrasyon | Deterministik Orkestratör |
| Ajan | Research Agent, Analysis & Preprocessing Agent, Evaluation & Self-Correction Agent |
| Çalışma | Arama/İndirme Araçları, Profilleme (pandas), Pipeline (scikit-learn), Değerlendirme (scikit-learn) |
| Kalıcı Depolama | Deneyim Belleği (SQLite), Veri Seti Deposu, Deney Kayıtları (SQLite) |

**Dış Servisler:** OpenML, UCI ML Repository, LLM API

### Önemli Tasarım Kararları
- **LLM yalnızca karar desteği sağlar.** Sayısal hesaplamalar (dönüşümler, model eğitimi, metrik hesaplama) Python araçlarıyla yürütülür.
- **Deterministik Orkestratör** ajanları yönetir; LLM orkestrasyon kararları vermez.

### Güncelleme
`npx @mermaid-js/mermaid-cli -i architecture.mmd -o architecture.svg -w 2200`

---

## Diyagram 3 — Veri Seti Keşfi ve Doğrulama

**Araç:** Mermaid | **Dosya:** `03-dataset-discovery/`

### Amaç
Research Agent'ın veri kaynağı arama, aday listeleme, güvenli indirme ve doğrulama sürecini, tüm hata senaryolarıyla birlikte gösterir.

### Normal Akış (10 adım)
Kullanıcı isteği → Research Agent yorumlama → OpenML+UCI arama → Aday listeleme → Kullanıcı seçimi → Güvenli indirme → Dosya doğrulama → Hash kontrolü → Depolama → Profilleme aşamasına geçiş

### Hata Senaryoları ve Çözüm Akışları
| Hata | Çözüm |
|------|-------|
| Veri kaynağına erişilemiyor | 3× yeniden dene → Erişilebilen kaynakla devam → Kullanıcı yükleme öner |
| Uygun aday bulunamıyor | Kriterleri genişlet → Kullanıcıdan farklı kriter iste |
| İndirme başarısız | 3× yeniden dene → Farklı aday seç |
| Desteklenmeyen dosya biçimi | Kullanıcıya bildir, farklı aday öner |
| Boyut sınırı aşıldı (>50K satır) | Kullanıcıya bildir, daha küçük aday öner |
| Dosya bozuk / okunamıyor | Hata kaydedilir, kullanıcıya bildirilir, alternatif önerilir |

### Güncelleme
`npx @mermaid-js/mermaid-cli -i dataset-discovery.mmd -o dataset-discovery.svg -w 2000`

---

## Diyagram 4 — Veri Profilleme ve Ön İşleme

**Araç:** Mermaid | **Dosya:** `04-data-preprocessing/`

### Amaç
Veri profilleme ve scikit-learn Pipeline oluşturma sürecini gösterir. **Veri sızıntısı önleme mekanizması** bu diyagramın araştırma açısından en kritik bileşenidir.

### Veri Sızıntısı Önleme Mekanizması
```
Veri seti yüklenir
        │
        ▼
🔒 Train / Test Ayrımı (% 80 / % 20)
   Test seti kilitlenir — strateji seçiminde kullanılmaz
        │
        ▼
Tüm profilleme YALNIZCA Eğitim Verisi üzerinde
        │
        ▼
scikit-learn Pipeline:
  fit(X_train) → Dönüşümler eğitimden öğrenilir
  transform(X_test) → Test verisi yalnızca dönüştürülür
```

### Profilleme Adımları (Yalnızca Eğitim Verisi)
Veri türleri → Eksik değer oranları → Kategorik değişkenler → Sayısal dağılımlar → Aykırı değerler → Veri profili

### Hata Senaryoları ve Çözüm Akışları
| Hata | Çözüm |
|------|-------|
| Veri seti boş | Hata kaydedilir, kullanıcıya bildirilir, işlem sonlandırılır |
| Hedef değişken bulunamıyor | Kullanıcıdan istenir; kullanıcı belirtmezse sistem otomatik tahmin dener |
| Desteklenmeyen veri türü | Sütun atlanır, uyarı loglanır |
| Uygun strateji üretilemiyor | Varsayılan Yöntem A uygulanır |
| Pipeline oluşturma hatası | Hata kaydedilir, sonraki adaya geçilir; aday kalmadıysa Yöntem A |
| Geçersiz sütun yapısı | Kural tabanlı düzeltme veya sütun analiz dışı bırakılır |

### Güncelleme
`npx @mermaid-js/mermaid-cli -i preprocessing.mmd -o preprocessing.svg -w 2000`

---

## Diyagram 5 — Değerlendirme, Özdenetim ve Deneyim Belleği

**Araç:** Mermaid | **Dosya:** `05-evaluation-self-correction/`

### Amaç
Projenin araştırma çekirdeğini oluşturan Evaluation & Self-Correction Agent'ın değerlendirme ve özdenetim döngüsünü, deneyim belleği entegrasyonunu ve kontrol mekanizmalarını gösterir.

### Kontrol Mekanizmaları
| Mekanizma | Kural |
|-----------|-------|
| Deneme sınırı | En fazla 5 deneme |
| Erken durdurma | Art arda 2 başarısız denemede durur |
| Süre sınırı | Belirlenen süre dolduğunda durur |
| Tekrar engeli | Daha önce denenen strateji tekrar denenmez |
| LLM hata yönetimi | Geçersiz çıktıda kural tabanlı varsayılan strateji devreye girer |

### Deneyim Belleği Kullanımı
- **Önce:** Benzer veri profili için geçmişte başarılı stratejiler sorgulanır (o veri setine ait kayıtlar dışlanır).
- **Sonra:** Her denemenin sonucu (başarılı veya başarısız) belleğe kaydedilir.

### Test Seti Güvenliği
Test seti yalnızca tüm denemeler tamamlandıktan sonra, **bir kez** açılır. Çapraz doğrulama yalnızca eğitim verisi üzerinde yapılır.

### Hata Senaryoları ve Çözüm Akışları
| Hata | Çözüm |
|------|-------|
| Model eğitimi başarısız | Hata kaydedilir, yeniden deneme hakkı varsa devam, yoksa en iyiyle sonlandır |
| Geçersiz performans sonucu | Strateji geçersiz işaretlenir, aynı yol |
| LLM geçersiz strateji öneriyor | Hata kaydedilir, kural tabanlı varsayılan devreye girer |
| Daha önce denenen strateji önerisi | Atlanır, yeni öneri istenir |
| Deneme sınırı doldu | En iyi geçerli stratejiyle sonlandırılır |
| Süre sınırı aşıldı | En iyi geçerli stratejiyle sonlandırılır |

### Güncelleme
`npx @mermaid-js/mermaid-cli -i evaluation.mmd -o evaluation.svg -w 2000`

---

## Diyagram 6 — Uçtan Uca Yapay Zeka Pipeline

**Araç:** Mermaid (LR yönü) | **Dosya:** `06-ai-pipeline/`

### Amaç
Kullanıcı isteğinden sonuç raporuna kadar tüm yapay zeka ve makine öğrenmesi iş akışını tek diyagramda gösterir.

### Pipeline Aşamaları (8 aşama)
| Aşama | Aktör Türü | Araç / Teknoloji |
|-------|-----------|-----------------|
| 1. İstek ve Veri Seti Keşfi | LLM (karar) + Python (arama) | Research Agent, OpenML/UCI API |
| 2. İndirme ve Doğrulama | Python araçları | HTTPS, hash kontrolü |
| 3. Veri Profilleme | Python araçları | pandas (yalnızca eğitim) |
| 4. Strateji Seçimi | LLM (karar) + Python (eleme) | Analysis & Preprocessing Agent |
| 5. Ön İşleme | Python araçları | scikit-learn Pipeline (fit → yalnızca eğitim) |
| 6. Model Değerlendirme | Python araçları | scikit-learn, 5-katlı CV |
| 7. Özdenetim Döngüsü | LLM (karar) | Evaluation & Self-Correction Agent |
| 8. Nihai Değerlendirme | Python araçları | scikit-learn (test seti — 1 kez) |

### Temel Ayrımlar
- **LLM kullanım noktaları:** Research Agent, Analysis & Preprocessing Agent, Evaluation Agent (karar ve gerekçe için)
- **Python araç kullanım noktaları:** Tüm veri dönüşümleri, model eğitimi, metrik hesaplama
- **Özdenetim geri bildirim döngüsü:** Aşama 7'de gösterilmiştir; başarısız strateji → LLM'e yeni strateji iste → Aşama 5'e geri dön
- **Deneyim Belleği:** Aşama 4'te strateji önceliği için okunur, Aşama 7'de sonuç kaydedilir

### Güncelleme
`npx @mermaid-js/mermaid-cli -i ai-pipeline.mmd -o ai-pipeline.svg -w 3200`

---

## Araçlar ve Ortam

| Araç | Versiyon | Kullanım |
|------|---------|---------|
| PlantUML | 1.2026.9 | Use Case diyagramı (D1) |
| Mermaid CLI (`@mermaid-js/mermaid-cli`) | 11.17.1 | D2–D6 Mermaid diyagramları |
| Java | Sistem varsayılanı | PlantUML çalışma zamanı |
| Node.js | 20.20.2 | Mermaid CLI çalışma zamanı |

## Notlar

- Kaynak dosyalar (`.puml`, `.mmd`) düzenlenebilir metin formatındadır.
- SVG'ler kaynak dosyalardan yeniden üretilebilir.
- Türkçe karakter desteği her iki araçta da test edilmiştir.
- Tüm bileşen isimleri diyagramlar arasında tutarlı tutulmuştur.
- Varsayımlar ve belirsizlikler için: [`assumptions.md`](assumptions.md)
