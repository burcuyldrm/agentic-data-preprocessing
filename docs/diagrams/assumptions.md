# Varsayımlar ve Belirsizlikler

Bu dosya, diyagram tasarımı sırasında proje öneri formunda açıkça belirtilmeyen konularda yapılan varsayımları belgelemektedir.

---

## Teknik Altyapı

| Konu | Varsayım | Kaynak |
|------|----------|--------|
| Web arayüzü teknolojisi | Teknoloji öneride henüz belirlenmemiş; diyagramlarda "Web Arayüzü (Frontend — teknoloji TBD)" olarak gösterilmiştir. | README, öneri formu |
| LLM sağlayıcısı ve modeli | Öneride "LLM API (sağlayıcı/model henüz seçilmedi)" ifadesi kullanılmıştır; diyagramlarda yalnızca "LLM API" olarak gösterilmiştir. | README |
| Backend görev kuyruğu | FastAPI arka plan görevleri (background tasks) veya benzeri bir mekanizma varsayılmıştır. Ayrı bir mesaj kuyruğu (Celery, RabbitMQ vb.) eklenmemiştir; öneride bu tür bir altyapıdan söz edilmemektedir. | Öneri formu |
| Kimlik doğrulama | JWT tabanlı kimlik doğrulama varsayılmıştır (FR-01, NFR-04 gereksinimlerinden). Öneride spesifik teknoloji belirtilmemiştir. | FR-01, NFR-04 |

---

## Proje Kapsamı

| Konu | Varsayım |
|------|----------|
| Veri formatı | Yalnızca CSV desteklenmektedir (öneride belirtilmiş). XLSX ve Hugging Face desteği zaman kalırsa değerlendirilecektir; bu formatlar diyagramlara dahil edilmemiştir. |
| Problem türleri | Yalnızca sınıflandırma ve regresyon. Kümeleme, zaman serisi vb. kapsam dışıdır. |
| Kullanıcı yükleme | Öneri formunda risk azaltma B-planı olarak belirtilmiştir. Use Case diyagramında "Veri Seti Yükle (Alternatif)" olarak gösterilmiştir. |
| Boyut sınırı | ≤50.000 satır (öneri formundan). Dataset Discovery akış diyagramında bu sınır kontrol adımı olarak gösterilmiştir. |

---

## Deney Tasarımı

| Konu | Varsayım |
|------|----------|
| A/B/C/D yaklaşımları | Use Case diyagramına dahil edilmemiştir; bunlar sistem içi deney konfigürasyonlarıdır, kullanıcıya sunulan farklı senaryolar değildir. System Architecture diyagramında Deterministik Orkestratör bu modu yönetir. |
| Tekrarlayan tohum değerleri | Ortak veri bölmeleri ve 5 farklı rastgelelik tohumu (öneri formu). Diyagramlarda tekrarlanabilirlik mekanizması genel olarak gösterilmiştir. |

---

## Veri Sızıntısı Önleme

| Konu | Yaklaşım |
|------|----------|
| Train/Test ayrımı noktası | Veri seti yüklendikten sonra, herhangi bir profilleme veya dönüşümden önce yapılır. Diyagram 4'te bu adım görsel olarak öne çıkarılmıştır. |
| scikit-learn Pipeline | `fit` yalnızca eğitim verisi üzerinde çağrılır; test verisi yalnızca `transform` ile dönüştürülür. Bu mekanizma Diyagram 4 ve 5'te açıkça gösterilmiştir. |
| Deneyim Belleği güvenliği | Bellek sorgulanırken o veri setine ait kayıtlar dışlanır; hash kontrolü ile aynı veri setinin farklı adlarla tekrar kullanılması engellenir. Bu kural Diyagram 5'te gösterilmiştir. |

---

## Proje Öneri Formu Hakkında Not

`project-proposal.docx` dosyası, geçerli bir Office XML/ZIP formatında değil, UTF-8 kodlamalı düz metin olarak kaydedilmiştir. İçerik, `README.md` ile büyük ölçüde örtüşmektedir. Diyagramlar her iki kaynaktaki bilgilere dayanılarak hazırlanmıştır.
