# Logistic Regression Projesi

Bu projede, yetişkin gelir tahmini problemi kapsamında, kişilerin yüksek gelir (>50K) veya düşük gelir (<=50K) sınıflarına ait olup olmadığını belirlemek için Logistic Regression algoritması kullanılmıştır. Uygulama iki farklı şekilde gerçekleştirilmiştir. Birinci yöntem, Scikit-Learn kütüphanesi ile hazır Logistic Regression modeli kullanılarak yapılmıştır. İkinci yöntem ise Python ile elle yazılmış (manual) Logistic Regression algoritması uygulanarak gerçekleştirilmiştir.

## Veri Seti ve Ön İşleme

- **Veri Seti:**  
  Adult Income veri seti (UCI Machine Learning Repository) kullanılmıştır.  
  - Veri seti, demografik bilgiler (yaş, eğitim, çalışma durumu vb.) içeren ve kişilerin gelir durumunu belirten örneklerden oluşmaktadır.
  - Toplam örnek sayısı 6500 civarında olup, yeterli özellik ve örnek bulunmaktadır.

- **Özellikler ve Dönüşüm:**  
  - Ham veri, kategorik özelliklerin modelde kullanılabilmesi için one-hot encoding yöntemi ile sayısallaştırılmıştır.
  - Sayısal özellikler, StandardScaler ile ölçeklendirilmiştir.
  - Veri seti, stratified train-test split yöntemi kullanılarak eğitim ve test verilerine ayrılmıştır.

## Model Uygulaması

### 1. Scikit-Learn Logistic Regression

- **Uygulama:**  
  Scikit-Learn’ün `LogisticRegression` sınıfı kullanılarak model eğitilmiştir.

- **Performans Ölçümleri:**  
  - **Eğitim Süresi:** 0.1402 saniye  
  - **Test Süresi:** 0.0010 saniye  
  - **Accuracy:** %85.51  
  - **Karmaşıklık Matrisi:**  
    - [4598 doğru negatif, 347 yanlış pozitif]  
    - [597 yanlış negatif, 971 doğru pozitif]  
  - **Sınıflandırma Raporu:**  
    Yeniden hesaplanan precision, recall ve f1-score değerleri her sınıf için raporlanmıştır.

### 2. Manual (Elle Yazılmış) Logistic Regression

- **Uygulama:**  
  Python kullanılarak sıfırdan, maksimum likelihood estimation tabanlı cost function ve gradient descent algoritması ile Logistic Regression modeli oluşturulmuştur.

- **Performans Ölçümleri:**  
  - **Eğitim Süresi:** 2.6917 saniye  
  - **Test Süresi:** 0.0000 saniye  
  - **Accuracy:** %84.46  
  - **Karmaşıklık Matrisi:**  
    - [4522 doğru negatif, 423 yanlış pozitif]  
    - [589 yanlış negatif, 979 doğru pozitif]  
  - **Sınıflandırma Raporu:**  
    Modelin performansı detaylı olarak raporlanmıştır.

## Karşılaştırma ve Sonuçlar

- **Doğruluk Oranı:**  
  Her iki model de yüksek doğruluk oranları (%85 civarında) elde etmiştir. Scikit-Learn modeli ile elle yazılan model arasında yaklaşık %1’lik bir fark bulunmaktadır.

- **Eğitim ve Test Süreleri:**  
  - Scikit-Learn modeli, optimize edilmiş algoritmaları sayesinde çok daha kısa sürede eğitilmiş ve tahmin yapmıştır.
  - Elle yazılmış model, eğitim süresi açısından daha uzun çalışmış ancak algoritmanın matematiksel mantığını kavramak açısından faydalı bir deneyim sunmuştur.

- **Karmaşıklık Matrisi ve Diğer Metrikler:**  
  İki modelin karmaşıklık matrisleri ve sınıflandırma raporları oldukça benzerdir. Bu da, her iki yöntemin de benzer performans sergilediğini göstermektedir.

- **Tartışma:**  
  Scikit-Learn modelinin, ek optimizasyon teknikleri ve yerleşik regularization gibi özelliklerle pratikte çok daha hızlı sonuç verdiği görülmüştür. Ancak, manuel uygulama, algoritmanın temel prensiplerini öğrenmek ve modelin işleyişini daha iyi kavramak için değerli bir yaklaşımdır.

## Sonuç

Her iki yöntemde de benzer başarımlara ulaşılmış olup, pratik uygulamalarda Scikit-Learn modeli tercih edilirken, teorik öğrenim açısından manuel uygulama önemli bir referans sağlamaktadır. Proje, veri ön işleme, model eğitimi, performans ölçümü ve sonuçların karşılaştırılması adımlarını kapsamlı bir şekilde ele almıştır.


