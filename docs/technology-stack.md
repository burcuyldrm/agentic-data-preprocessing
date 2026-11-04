# Teknoloji Yığını

**Proje:** Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip Çok Ajanlı Otonom Veri Analizi ve Ön İşleme Sistemi  
**Geliştirici:** Burcu Yıldırım  
**Güncelleme:** 2026-11-05

---

## Durum Tanımları

| Durum | Açıklama |
|---|---|
| **Kesinleşmiş** | Proje öneri formunda veya README'de açıkça belirtilmiş ya da mevcut mimari kararlarla kilitlenmiş |
| **Önerilen** | Proje gereksinimlerine uygun, ancak henüz resmi olarak onaylanmamış |
| **Kararlaştırılmamış** | Proje dokümanlarında "TBD" veya "henüz seçilmedi" olarak geçiyor |

---

## Teknoloji Yığını Tablosu

| Katman | Teknoloji | Kullanım Amacı | Tercih Gerekçesi | Alternatif | Durum |
|---|---|---|---|---|---|
| **Programming Language** | Python 3.11+ | Tüm sistem bileşenleri: ajanlar, orkestratör, araçlar | Geniş ML/veri ekosistemi; LLM SDK desteği; tek geliştirici için yeterli | — | Kesinleşmiş |
| **Backend Framework** | FastAPI | REST API, endpoint yönetimi, arka plan görevleri | Async destek, tip doğrulama, otomatik OpenAPI/Swagger; proje dokümanında belirtilmiş | Flask, Django | Kesinleşmiş |
| **Authentication** | JWT (python-jose veya PyJWT) | Kullanıcı kimlik doğrulama; `/analyze`, `/results` gibi korumalı endpointler | Durumsuz; FR-01 ve NFR-04 gereksinimlerinden türetildi | Session-based auth | Önerilen |
| **Database** | SQLite 3.25+ | Deneyim belleği, deney kayıtları, kullanıcı hesapları | Hafif, sıfır konfigürasyon, sıfır ek servis; proje öneri formunda belirtilmiş | PostgreSQL | Kesinleşmiş |
| **Database Access** | Python `sqlite3` (standart kütüphane) veya SQLAlchemy Core | Veritabanı bağlantı yönetimi, PRAGMA ayarları | `sqlite3` bu ölçek için yeterli; SQLAlchemy bağlantı havuzu gerekirse tercih edilir | raw SQL, SQLAlchemy ORM | Kararlaştırılmamış |
| **Data Processing** | pandas 2.x | Veri profilleme, istatistik hesaplama, CSV okuma | NumPy tabanlı; ML ekosisteminin standardı; proje öneri formunda belirtilmiş | Polars | Kesinleşmiş |
| **Numerical Computing** | NumPy | Matris işlemleri, dizin hesaplamaları | pandas ve scikit-learn'ün alt yapısı; bağımlılık olarak zaten yüklü | — | Kesinleşmiş |
| **Preprocessing / ML Pipeline** | scikit-learn | `Pipeline`, `ColumnTransformer`, impütasyon, ölçekleme, kodlama | Üretim kanıtlı; `fit`/`transform` ayrımıyla veri sızıntısı önleme mekanizması; proje öneri formunda belirtilmiş | — | Kesinleşmiş |
| **Model Evaluation** | scikit-learn (cross_validate, metrics) | 5 katlı çapraz doğrulama, macro-F1, balanced accuracy, RMSE, MAE | scikit-learn ile entegre; referans model (LogisticRegression, Ridge) desteği | — | Kesinleşmiş |
| **Dataset Discovery** | openml (Python paketi) | OpenML veri seti araması ve indirmesi | Resmi Python SDK; meta veri sorgulama ve otomatik indirme desteği | — | Kesinleşmiş |
| **Dataset Discovery** | UCI ML Repository (HTTP/ucimlrepo) | UCI veri seti araması ve indirmesi | Geniş arşiv; `ucimlrepo` paketi doğrudan erişim sağlar | Manuel HTTP indirme | Kesinleşmiş |
| **AI / LLM** | LLM API (sağlayıcı seçilmedi) | Ajan karar desteği: strateji seçimi, gerekçe üretimi, hata analizi | Proje öneri formunda "sağlayıcı/model henüz seçilmedi" | Anthropic Claude (Sonnet), OpenAI GPT-4o, Google Gemini | Kararlaştırılmamış |
| **Agent Orchestration** | Deterministik Python (özel orkestratör) | Ajan sırası, hata yönetimi, deneme sınırı, süre kontrolü | LLM orkestrasyon kararları vermez; öngörülebilir davranış, kolay hata ayıklama; proje tasarımı gereği | LangGraph, CrewAI, AutoGen | Kesinleşmiş |
| **Frontend** | Laravel (PHP) | Web arayüzü: kullanıcı etkileşimi, analiz başlatma, sonuç görüntüleme; FastAPI'ye HTTP istekleri gönderir | Blade şablon motoru ile sunucu taraflı render; entegre auth scaffold; hızlı UI geliştirme | Streamlit, Next.js + React | Kesinleşmiş |
| **Containerization** | Docker | Dağıtım, ortam tekrarlanabilirliği | Tüm bağımlılıkları sarar; proje öneri formunda belirtilmiş | Podman | Kesinleşmiş |
| **Testing** | pytest | Birim testleri, entegrasyon testleri, veritabanı doğrulama | Python standardı; fixture ve parametrize desteği; proje öneri formunda belirtilmiş | unittest | Önerilen |
| **Version Control** | Git + GitHub | Kod yönetimi, dal bazında geliştirme | Projede aktif olarak kullanılıyor | — | Kesinleşmiş |
| **Documentation — Diagrams** | Mermaid CLI 11.17.1 | ER diyagramı, akış diyagramları (D2–D6), mimari diyagram | Kod olarak yönetilebilir, SVG çıktısı; projede aktif olarak kullanılıyor | PlantUML | Kesinleşmiş |
| **Documentation — UML** | PlantUML 1.2026.9 | Use Case diyagramı (D1) | UML 2.5 standardına uygun; projede aktif olarak kullanılıyor | Mermaid | Kesinleşmiş |

---

## Kapsam Dışı Teknolojiler

Aşağıdaki teknolojiler proje öneri formunda açıkça kapsam dışı bırakılmıştır:

- Görüntü ve metin veri işleme kütüphaneleri (PyTorch, Transformers vb.)
- Derin öğrenme çerçeveleri (TensorFlow, Keras)
- Dağıtık büyük veri altyapıları (Spark, Dask)
- Pekiştirmeli öğrenme kütüphaneleri
- XLSX ve Hugging Face desteği (zaman kalırsa değerlendirilecek)

---

## Belirsiz Kalan Kararlar

| Karar | Seçenekler | Etki |
|---|---|---|
| LLM sağlayıcısı ve modeli | Anthropic Claude Sonnet, OpenAI GPT-4o, Google Gemini | API maliyeti, bağlam penceresi, Python SDK kalitesi |
| Veritabanı erişim katmanı | raw `sqlite3`, SQLAlchemy Core | Kod karmaşıklığı; bu ölçekte ikisi de yeterli |
