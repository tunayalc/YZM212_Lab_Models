### Logistic Regression Projesi

Bu projede, yetişkinlerin gelir durumunu (yüksek gelir >50K ya da düşük gelir <=50K) tahmin etmek için Logistic Regression algoritmasını kullanarak bir model geliştirdim. Projede iki farklı yaklaşım uyguladım: biri Scikit-Learn kütüphanesinin hazır modeli ile, diğeri ise Python kullanarak elle yazdığım (manual) Logistic Regression algoritması ile.

---

### Veri Seti ve Ön İşleme

**Veri Seti:**

- UCI Machine Learning Repository’den alınan Adult Income veri setini kullandım.
- Veri seti, yaş, eğitim, çalışma durumu gibi demografik bilgiler içeriyor.
- Toplam yaklaşık 6500 örnek mevcut; bu, modelleme için yeterli sayıda örnek sağlıyor.

**Ön İşleme Adımları:**

- **Kategorik Verilerin Dönüştürülmesi:** Kategorik değişkenler, modelde kullanılabilmesi için one-hot encoding yöntemiyle sayısallaştırıldı.
- **Özellik Ölçeklendirme:** Sayısal veriler, StandardScaler ile ölçeklendirildi.
- **Veri Setinin Bölünmesi:** Stratified train-test split yöntemi kullanılarak, veri dengeli bir şekilde eğitim ve test setlerine ayrıldı.

---

### Model Uygulamaları

#### 1. Scikit-Learn Logistic Regression

- **Uygulama:**  
  Scikit-Learn’ün `LogisticRegression` sınıfı ile modelimi oluşturdum ve eğittim.

- **Performans Ölçümleri:**
  - **Eğitim Süresi:** 0.1402 saniye
  - **Test Süresi:** 0.0010 saniye
  - **Doğruluk Oranı (Accuracy):** %85.51

- **Karmaşıklık Matrisi:**  
  \[
  \begin{bmatrix}
  4598 & 347 \\
  597 & 971 \\
  \end{bmatrix}
  \]
  
- **Sınıflandırma Raporu:**  
  Her sınıf için precision, recall ve f1-score değerleri detaylı olarak hesaplandı.

#### 2. Manual (Elle Yazılmış) Logistic Regression

- **Uygulama:**  
  Python kullanarak sıfırdan, maksimum likelihood estimation tabanlı cost function ve gradient descent algoritmasıyla modelimi oluşturdum.

- **Performans Ölçümleri:**
  - **Eğitim Süresi:** 2.6917 saniye
  - **Test Süresi:** 0.0000 saniye
  - **Doğruluk Oranı (Accuracy):** %84.46

- **Karmaşıklık Matrisi:**  
  \[
  \begin{bmatrix}
  4522 & 423 \\
  589 & 979 \\
  \end{bmatrix}
  \]

- **Sınıflandırma Raporu:**  
  Modelin performansı detaylı metriklerle raporlandı.

---

### Karşılaştırma ve Sonuçlar

- **Doğruluk Oranı:**  
  İki model de yaklaşık %85 doğruluk oranı elde etti. Scikit-Learn modelinde %85.51, manuel modelde ise %84.46 gibi benzer sonuçlar görüldü.

- **Eğitim ve Test Süreleri:**  
  Scikit-Learn modeli, optimize edilmiş algoritmalar sayesinde çok daha hızlı eğitim aldı ve test edildi. Elle yazdığım model ise eğitim süresi bakımından daha uzun sürdü, fakat algoritmanın temel prensiplerini öğrenmek açısından oldukça faydalı oldu.

- **Karmaşıklık Matrisi ve Diğer Metrikler:**  
  Her iki modelin karmaşıklık matrisleri ve sınıflandırma raporları birbirine oldukça yakın performans sergiliyor.

---

### Tartışma

Scikit-Learn’ün hazır Logistic Regression modeli, optimize edilmiş algoritmaları ve ek regularization gibi özellikleri sayesinde pratik uygulamalarda tercih ediliyor. Ancak, manuel olarak yazdığım Logistic Regression modelini geliştirirken algoritmanın matematiksel temellerini daha iyi kavradım ve bu süreç, teorik bilgimin pekişmesine yardımcı oldu. İki yöntemin de sonuçlarının birbirine yakın çıkması, doğru veri ön işleme ve model kurulumunun önemini ortaya koydu.

---

### Sonuç

Her iki yaklaşım da başarılı sonuçlar verdi. Pratik uygulamalarda Scikit-Learn modelinin daha hızlı ve etkili olduğu görülürken, teorik öğrenim açısından manuel uygulama önemli bir referans oldu. Projede, veri ön işleme, model eğitimi, performans ölçümü ve sonuçların karşılaştırılması gibi adımları detaylıca ele aldım.

---

Bu rapor, Logistic Regression algoritmasını hem uygulamalı hem de teorik açıdan değerlendirmemi sağlayarak, konuyla ilgili kapsamlı bir bakış açısı sunmaktadır.
