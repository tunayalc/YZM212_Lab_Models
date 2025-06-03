# Sigorta Ücretleri Tahmini Projesi

## Proje Özeti
Bu projede, sağlık sigortası ücretlerini tahmin etmek için tek gizli katmanlı bir sinir ağı (neural network) sıfırdan (NumPy ve Pandas kullanarak) uygulanmıştır. Amaç, TensorFlow, Keras veya PyTorch gibi hazır kütüphaneler kullanılmadan bir modelin nasıl eğitildiğini adım adım göstermek, eğitim kaybı ve test sonuçlarını grafiklerle görselleştirmektir.

Özellikle:
- Kategorik değişkenlerin tek-sıcak kodlama (one-hot encoding) ile sayısallaştırılması,
- Min–Max ölçekleme ile özellik ve hedef değerlerin normalizasyonu,
- Tam küme (batch) gradyan inişi (gradient descent) ile ileri ve geri yayılımın (forward/backward propagation) elle uygulanması,
- Eğitim kaybı (MSE) ve test performansının görselleştirilmesi.

## Veri Seti ve Ön İşleme

- **Veri Seti:**  
  “insurance.csv” dosyası (Kaggle’daki “Medical Cost Personal Datasets”) kullanılmıştır.  
  - Toplam 1.338 örnek içerir.  
  - Kolonlar:  
    - age (tamsayı)  
    - sex (“female”/“male”)  
    - bmi (ondalık)  
    - children (tamsayı)  
    - smoker (“yes”/“no”)  
    - region (“northeast”, “northwest”, “southeast”, “southwest”)  
    - charges (sigorta ücreti, ondalık) ← Hedef değişken

- **Kategorik Değişkenlerin Kodlanması:**  
  - `sex` → female: 0, male: 1  
  - `smoker` → no: 0, yes: 1  
  - `region` → dört ayrı tek-sıcak kod sütunu:  
    - region_northeast, region_northwest, region_southeast, region_southwest  

- **Öznitelik ve Hedef Normalizasyonu (Min–Max Ölçekleme):**  
  Tüm sayısal öznitelikler (age, sex, bmi, children, smoker, region_* ) ve hedef değişken (charges) 0–1 aralığına ölçeklenir:  
  ```
  x_scaled = (x – x_min) / (x_max – x_min)
  ```
  Eğitim kümesinde hesaplanan min ve max değerleri, test verisine de aynı şekilde uygulanır.

- **Eğitim/Test Ayrımı:**  
  - %80 eğitim, %20 test  
  - Karıştırma (shuffle) işlemi sabit bir random seed ile yapılır (tekrarlanabilirlik için).  

## Model Mimarisi
- **Girdi Katmanı:**  
  - Boyut: 8 öznitelik (age, sex, bmi, children, smoker, region_northeast, region_northwest, region_southeast, region_southwest)  
- **Gizli Katman:**  
  - 16 nöron  
  - Aktivasyon: ReLU (`ReLU(z) = max(0, z)`)  
  - Ağırlıklar (W1): Normal dağılım (μ=0, σ=0.01) ile başlatılır  
  - Eğimsiz sap (b1): Sıfır vektörü  
- **Çıktı Katmanı:**  
  - 1 nöron (sigorta ücretini tahmin eder)  
  - Aktivasyon: Lineer (regresyon için)  
  - Ağırlıklar (W2): Normal dağılım (μ=0, σ=0.01)  
  - Eğimsiz sap (b2): Sıfır vektörü  

Toplam parametre matrisi boyutları:  
- W1: (8 × 16), b1: (1 × 16)  
- W2: (16 × 1), b2: (1 × 1)  

## Eğitim Süreci
- **Kayıp Fonksiyonu:**  
  Mean Squared Error (MSE):  
  ```
  MSE = (1/m) * Σ (ŷᵢ – yᵢ)²
  ```
  Burada, m eğitim örnek sayısıdır, ŷᵢ modelin tahmin değeri, yᵢ gerçek değerdir.

- **Optimizasyon:**  
  Tam küme (batch) gradyan inişi (gradient descent) kullanılır.  
  - Öğrenme hızı (α): 0.01  
  - Epoch sayısı: 1.000  
  - Batch boyutu: Eğitim setinin tamamı (m = n_egitim örnek)  

- **Her Epoch İçin Adımlar:**  
  1. **İleri Yayılım (Forward Pass):**  
     - Gizli katman:  
       ```
       Z¹ = X · W1 + b1  
       A¹ = ReLU(Z¹)
       ```  
     - Çıktı:  
       ```
       Z² = A¹ · W2 + b2  
       ŷ = Z²   (lineer aktivasyon)
       ```
  2. **Kayıp Hesabı:**  
     ```
     L = (1/m) * Σ (ŷᵢ – yᵢ)²
     ```
  3. **Geri Yayılım (Backward Pass):**  
     - Çıktı katmanı türevleri:  
       ```
       dZ² = (2/m) * (ŷ – y)  
       dW2 = (A¹)ᵀ · dZ²  
       db2 = Σ dZ²  
       ```
     - ReLU geri yayılımı:  
       ```
       dA¹ = dZ² · (W2)ᵀ  
       dZ¹ = dA¹ * ReLU’(Z¹)   (ReLU’(z) = 1 eğer z>0, aksi hâlde 0)  
       dW1 = Xᵀ · dZ¹  
       db1 = Σ dZ¹  
       ```
  4. **Parametre Güncelleme:**  
     ```
     W1 ← W1 – α * dW1  
     b1 ← b1 – α * db1  
     W2 ← W2 – α * dW2  
     b2 ← b2 – α * db2  
     ```

- **Eğitim Sırasında Çıktılar:**  
  - Her 100 epoch’ta bir eğitim kaybı (normalized MSE) konsola yazdırılır, örnek çıktılar:  
    ```
    Epoch 0, Loss: 0.4922  
    Epoch 100, Loss: 0.4913  
    Epoch 200, Loss: 0.4882  
    ...  
    Epoch 900, Loss: 0.1243  
    ```
  - Sonuç olarak normalize edilmiş MSE değerleri hem eğitim hem test verisi için raporlanır.

## Sonuçlar

1. **Korelasyon Matrisi (Isı Haritası)**  
   - Özellikler arasındaki Pearson korelasyon katsayıları görselleştirilir.  
   - Özellikle `smoker` ve `charges` arasındaki korelasyon ~0.79 ile yüksektir.  
   - `bmi` ve `charges` arasında orta düzeyde pozitif korelasyon (~0.20) gözlemlenir.

2. **Eğitim Kaybı (Epoch vs. MSE)**  
   - 1.000 epoch boyunca, başlangıçta (Epoch 0) normalize MSE ≈ 0.4922 idi.  
   - Epoch 1.000 sonunda eğitim kaybı ≈ 0.12’ye düştü (normalized).  

3. **Test Performansı (Gerçek vs. Tahmin Dağılımı)**  
   - Test kümesi için gerçek sigorta ücretleri x ekseninde, modelin tahmin değerleri y ekseninde gösterildi.  
   - Kırmızı kesikli çizgi (y = x) ideal tahmini temsil eder; noktalar bu çizgiye ne kadar yakınsa o kadar iyi tahmin yapılmış demektir.

4. **Final MSE Değerleri (Normalize Edilmiş):**  
   - Eğitim MSE: ≈ 0.2381  
   - Test MSE: ≈ 0.2196  

5. **Örnek Tahmin Sonuçları (Test Kümesi İlk 10 Örnek):**  
   | İndeks | Gerçek Ücret | Tahmin Edilen Ücret |  
   |--------|--------------|---------------------|  
   | 0      | 9095.06825   | 8083.431183         |  
   | 1      | 5272.17580   | 7231.679608         |  
   | 2      | 29330.98315  | 35380.288919        |  
   | 3      | 9301.89355   | 8210.185429         |  
   | 4      | 33750.29180  | 27026.498442        |  
   | 5      | 4536.25900   | 11652.276124        |  
   | 6      | 2117.33885   | 5900.253956         |  
   | 7      | 14210.53595  | 17129.200207        |  
   | 8      | 3732.62510   | 6386.994884         |  
   | 9      | 10264.44210  | 8877.911203         |  

## Kaynakça
- Kaggle – Medical Cost Personal Datasets: https://www.kaggle.com/mirichoi0218/insurance  
- (https://devhunteryz.wordpress.com/2018/06/20/geri-yayilimbackpropagation/)
