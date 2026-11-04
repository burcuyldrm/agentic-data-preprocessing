Agentic Data Preprocessing
Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip Çok Ajanlı Otonom Veri Analizi ve Ön İşleme Sistemi
Proje durumu: Planlama ve ilk geliştirme aşaması. Aşağıda açıklanan sistem özellikleri geliştirme hedefleridir; henüz tamamlanmış veya çalışır durumda oldukları iddia edilmemektedir.

Proje Hakkında
Makine öğrenmesi projelerinde eksik değerlerin doldurulması, kategorik değişkenlerin kodlanması ve sayısal özelliklerin ölçeklenmesi gibi ön işleme kararları, model performansını etkileyebilmektedir. Ancak her veri setine aynı ön işleme yöntemlerinin uygulanması uygun sonuçlar vermeyebilir.

Bu projede, tanımlanan analiz problemine uygun açık veri setlerini araştıran, verileri profilleyen, önceden doğrulanmış ön işleme stratejileri arasından seçim yapan ve seçimlerini model performansına göre güncelleyen LLM destekli çok ajanlı bir sistem geliştirilmesi amaçlanmaktadır.

Sistemin iki temel araştırma bileşeni bulunmaktadır:
- Performans geri bildirimi ve özdenetim: Ön işleme stratejilerinin başarısını ölçerek sınırlı sayıda alternatif deneme yapılması.
- Deneyim belleği: Önceki veri setlerinin profillerini, denenen stratejileri, performans sonuçlarını, çalışma sürelerini ve hataları saklayarak yeni veri setlerinde aday stratejilerin önceliklendirilmesi.
Temel araştırma sorusu, bu iki mekanizmanın sabit bir deneme bütçesi altında model başarısını artırıp artırmadığı ve/veya gerekli deneme sayısını azaltıp azaltmadığıdır.
Önerilen Sistem Mimarisi
Kullanıcı isteği / Web arayüzü
              |
              v
FastAPI + Deterministik Orkestratör
              |
              v
       Research Agent
    (OpenML / UCI arama)
              |
              v
 Analysis & Preprocessing Agent
   (veri profili ve strateji)
              |
              v
Evaluation & Self-Correction Agent
   (doğrulama ve geri bildirim)
              |                 ^
              +-- yeniden dene --+  (en fazla 5 deneme)
              |
              v
       Deneyim Belleği
           (SQLite)
              |
              v
    Son değerlendirme ve rapor
Ajanlar karar verme ve gerekçelendirme süreçlerini destekleyecektir. Veri dönüşümleri, model eğitimi ve metrik hesaplamaları serbest biçimde üretilen kodlarla değil, önceden tanımlanmış Python araçlarıyla yürütülecektir. Deterministik orkestratör; görev sırasını, hata yönetimini ve çalışma sınırlarını kontrol edecektir.
Bileşenlerin Görevleri
Bileşen	Planlanan görev
Research Agent	OpenML ve UCI üzerinde uygun veri setlerini aramak, adayları listelemek ve seçilen veriyi güvenli biçimde indirmek.
Analysis & Preprocessing Agent	Değişken türleri, eksik değer oranları, kategori sayıları, çarpıklık ve aykırı değerleri incelemek; uygun stratejiyi seçmek.
Evaluation & Self-Correction Agent	Aday ön işleme hatlarını değerlendirmek, performans geri bildirimine göre alternatif seçmek ve deneme sınırlarına uymak.
Deterministik Orkestratör	Ajanlar arası akışı, geçerlilik kontrollerini, hataları ve çalışma sınırlarını yönetmek.
Deneyim Belleği	Veri profillerini ve deney sonuçlarını SQLite üzerinde saklamak; geçmiş deneyimlere göre strateji adaylarını sıralamak.


Planlanan Özellikler
- OpenML ve UCI üzerinden veri seti araştırma; kaynak, dosya türü ve boyut doğrulaması.
- CSV biçimindeki tablosal veriler üzerinde sınıflandırma ve regresyon işlemleri.
- Veri profili çıkarma ve ön işleme kararlarının gerekçeleriyle raporlanması.
- Eksik değer doldurma, one-hot kodlama, standart ve robust ölçekleme yöntemlerinin scikit-learn Pipeline ile uygulanması.
- Aynı stratejiyi tekrarlamadan en fazla beş denemelik geri bildirim döngüsü.
- SQLite tabanlı deneyim belleği; geçmiş deneyimler aday sıralamasında kullanılacak, yeni veri setindeki performans ayrıca ölçülecektir.
- Seçilen strateji, denemeler, karar gerekçeleri ve nihai test sonuçlarını içeren raporlar.
- Kimlik doğrulamalı web arayüzü ve Docker ile dağıtım.
İlk sürümün kapsamı dışında: Görüntü ve metin verileri, derin öğrenme, dağıtık büyük veri altyapıları ve pekiştirmeli öğrenme. XLSX ve Hugging Face desteği zaman kalırsa değerlendirilecektir.
Deney Tasarımı
Dört yaklaşım kontrollü koşullarda karşılaştırılacaktır:
Yöntem	Açıklama
A – Sabit Ön İşleme	Ortanca/en sık değerle eksik veri doldurma, one-hot kodlama ve standart ölçekleme.
B – Tek Aşamalı Seçim	Veri profiline göre tek strateji seçimi; yeniden deneme yok.
C – Geri Bildirimli Seçim	Performans geri bildirimiyle en fazla beş farklı stratejinin değerlendirilmesi.
D – Deneyim Belleği Destekli Seçim	C ile aynı deneme bütçesi; stratejilerin önceki deneyimlere göre sıralanması.


Değerlendirme Protokolü
- Veri setleri: 6–10 açık erişimli tablosal veri seti.
- Test ayrımı: Başlangıçta ayrılan %20 test kümesi; strateji seçimine dahil edilmeyecektir.
- Doğrulama: Kalan veri üzerinde 5 katlı çapraz doğrulama.
- Referans modeller: Sınıflandırma için lojistik regresyon, regresyon için Ridge regresyon.
- Başarım ölçütleri: Sınıflandırmada makro-F1 ve balanced accuracy; regresyonda RMSE ve MAE.
- Tekrarlanabilirlik: Ortak veri bölmeleri ve beş farklı rastgelelik tohumu.
- Verimlilik: Deneme sayısı, çalışma süresi ve LLM çağrı sayısı.
- Veri sızıntısını önleme: Ön işleme dönüşümleri yalnızca eğitim katlarında öğrenilecek; test sonuçları strateji seçimine dahil edilmeyecektir. Bellek değerlendirmesinde ilgili veri setinin kayıtları dışlanacak ve tekrarları belirlemek için dosya özeti (hash) kontrolü kullanılacaktır.
Araştırma Hipotezleri
- H1: Geri bildirimli seçimin (C), sabit ön işleme (A) ve tek aşamalı seçime (B) göre birincil performans ölçütünde daha başarılı olması beklenmektedir.
- H2: Deneyim belleği destekli seçimin (D), aynı deneme bütçesinde C ile benzer veya daha yüksek başarıya daha az denemeyle ulaşması beklenmektedir.
Öngörülen kabul hedefleri (ölçülmüş sonuç değildir): Çalıştırmaların en az %95'inin başarıyla tamamlanması; veri setlerinin en az yarısında A'ya göre birincil ölçütte en az %2 göreli iyileşme; en fazla 50.000 satırlık veri setlerinde çalışma başına en çok 5 deneme, 10 dakika ve 15 LLM çağrısı. Olumsuz sonuçlar da değiştirilmeden raporlanacaktır.
Planlanan Teknoloji Yığını
Katman	Teknoloji
Programlama ve veri işleme	Python, pandas, scikit-learn
API ve iş akışı	FastAPI, deterministik Python orkestrasyonu
Ajan karar desteği	LLM API (sağlayıcı/model henüz seçilmedi)
Deneyim belleği	SQLite
Kullanıcı arayüzü	Web arayüzü (teknoloji henüz seçilmedi)
Dağıtım	Docker
Test ve kod kalitesi	pytest ve otomatik kontroller (kurulacak)


Geliştirme Yol Haritası
- [x] GitHub reposunun ve başlangıç README dosyasının oluşturulması.
- [ ] Kullanım senaryoları, sistem mimarisi, hata akışları ve diyagramların hazırlanması.
- [ ] Arayüz prototipi, veritabanı şeması ve veri sözlüğünün tasarlanması.
- [ ] Veri edinme, profilleme, ön işleme araçları ve A/B karşılaştırmalarının geliştirilmesi.
- [ ] Backend, kalıcı veri saklama, kimlik doğrulama ve API dokümantasyonunun hazırlanması.
- [ ] Frontend–backend entegrasyonu ve geri bildirimli C yönteminin geliştirilmesi.
- [ ] Deneyim belleği ve D yönteminin uygulanması.
- [ ] A–D deneylerinin tamamlanması ve karşılaştırmalı sonuçların raporlanması.
- [ ] Docker ile dağıtım; güvenlik, entegrasyon ve yük testlerinin yapılması.
- [ ] Kullanıcı kılavuzu, demo, final raporu ve akademik çalışma taslağının hazırlanması.
Kurulum ve Çalıştırma
Uygulanabilir ilk sürüm henüz geliştirilmemiştir. Çalışan veri işleme hattı repoya eklendiğinde kurulum, yapılandırma ve çalıştırma komutları bu bölümde paylaşılacaktır. Bu aşamada çalışan API, web arayüzü veya ajan akışı bulunduğu iddia edilmemektedir.
Geliştirme ve Tekrarlanabilirlik İlkeleri
Değişiklikler anlamlı commit mesajlarıyla izlenecek; uygun durumlarda ayrı geliştirme dalları ve Pull Request'ler kullanılacaktır. Deney yapılandırmaları, rastgelelik tohumları, bağımlılık sürümleri, veri setlerinin kaynak ve sürüm bilgileri ile sonuç metrikleri kaydedilecektir. API anahtarları, parolalar ve özel veri setleri Git deposuna eklenmeyecektir.
Akademik Bağlam
Bu proje, Yapay Zeka ve Veri Mühendisliği Tasarımı dersi kapsamında geliştirilmektedir.

Proje başlığı: Deneyim Belleği ve Özdenetim Mekanizmalarına Sahip Çok Ajanlı Otonom Veri Analizi ve Ön İşleme Sistemi

Geliştirici: Burcu Yıldırım
Lisans
Henüz bir açık kaynak lisansı seçilmemiştir. Lisans eklenene kadar bu repository için açık kaynak kullanım izni verilmiş sayılmaz. Herkese açık yayımlamadan önce uygun lisans belirlenecektir.