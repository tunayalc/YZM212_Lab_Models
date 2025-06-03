# Sigorta Ücretleri Tahmini (Sıfırdan İleri ve Geri Yayılım)

Bu depo, `insurance.csv` veri seti üzerinde yalnızca NumPy ve Pandas kullanarak sıfırdan bir sinir ağı (ileri ve geri yayılım) uygulamasını içerir. Amaç, hazır kütüphaneler (TensorFlow, Keras, PyTorch vb.) kullanmadan bir modelin nasıl eğitildiğini ve sonuçların nasıl görselleştirildiğini göstermektir.


## Proje Özeti

Bu laboratuvar ödevinde, **regresyon** problemi olan sigorta ücretlerini tahmin etmek için sıfırdan bir sinir ağı (tek gizli katmanlı) uygulanmaktadır. Model:

- Veri setini ön işler (eksik yok, kategorik değişkenleri tek-sıcak kodlama ile sayısallaştırır),
- Geri yayılım (backpropagation) ve Mean Squared Error (MSE) kayıp fonksiyonunu kullanarak tam küme (batch) gradyan inişiyle eğitilir,
- Eğitim kaybı (MSE) ve test sonuçlarını görselleştirir.

Üretilen çıktılar:
1. **Öznitelik Korelasyon Matrisi** (ısı haritası).  
2. **Eğitim Kayıp Grafiği** (Epoch’lara göre MSE).  
3. **Gerçek vs. Tahmin** dağılma grafiği (test kümesi).  
4. Basit örnek tahmin sonuçları ve son MSE değerleri.

---

## Veri Seti

Sigorta ücretleri verisi ("insurance.csv"), Kaggle üzerinden elde edilebilir:  
https://www.kaggle.com/mirichoi0218/insurance

Toplam 1.338 kayıt içerir ve aşağıdaki sütunlara sahiptir:

- `age` (yaş, tamsayı)  
- `sex` (cinsiyet, “male”/“female”)  
- `bmi` (vücut kitle indeksi, ondalık)  
- `children` (çocuk sayısı, tamsayı)  
- `smoker` (“yes”/“no”)  
- `region` (“northeast”, “northwest”, “southeast”, “southwest”)  
- `charges` (sigorta ücreti, ondalık) ← Hedef değişken  

### Örnek Kayıtlar

```
   age     sex     bmi  children smoker     region      charges
0   19  female  27.90         0    yes  southwest  16884.92400
1   18    male  33.77         1     no  southeast   1725.55230
2   28    male  33.00         3     no  southeast   4449.46200
3   33    male  22.70         0     no  northwest  21984.47061
4   32    male  28.88         0     no  northwest   3866.85520
```

---

## Gereksinimler

- **Python 3.7+**  
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **Seaborn**  

Kurulum için:
```bash
pip install numpy pandas matplotlib seaborn
```

---

## Depo Yapısı

```
.
├── insurance.csv
├── ForwardAndBackwardPropagation.py
├── README.md
├── plots/
│   ├── correlation_matrix.png
│   ├── loss_curve.png
│   └── actual_vs_predicted.png
└── requirements.txt
```

1. **insurance.csv**  
   Ham veri seti (1.338 örnek).  
2. **ForwardAndBackwardPropagation.py**  
   - Veriyi yükler ve ön işler.  
   - İleri ve geri yayılım adımlarını (MSE kaybı) sıfırdan uygular.  
   - Ağı 80% eğitim, 20% test olacak şekilde böler.  
   - Üç adet grafik oluşturur:  
     1. Korelasyon matrisi ısı haritası  
     2. Epoch’lara göre eğitim kayıp eğrisi  
     3. Test kümesi için gerçek vs. tahmin dağılım grafiği  
   - Eğitim kaybını her 100 epoch’ta bir ekrana yazdırır, son 1000 epoch’luk işlemi raporlar.  
3. **plots/**  
   Oluşturulan grafikler (PNG).  
4. **requirements.txt**  
   Gerekli paketlerin listesi.  

---

## Ön İşleme

1. **Veriyi Yükleme**  
   ```python
   import pandas as pd
   df = pd.read_csv("insurance.csv")
   ```
2. **Kategorik Kolonların Kodlanması**  
   - `sex` → `0` (female), `1` (male)  
   - `smoker` → `0` (no), `1` (yes)  
   - `region` → 4 adet tek-sıcak kod (one-hot) sütunu:  
     `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest`
3. **Öznitelik Matrisi ve Hedef**  
   - Öznitelikler (`X`):  
     `age`, `sex`, `bmi`, `children`, `smoker` ve 4 adet region dummy.  
   - Hedef (`y`): `charges` (sürekli).  
4. **Özellik ve Hedef Normalizasyonu (Min-Max Ölçekleme)**  
   \[
   x_{	ext{scaled}} = rac{x - x_{\min}}{x_{\max} - x_{\min}}
   \]
   Tüm öznitelik ve hedef, 0–1 aralığına ölçeklenir. Aynı faktörler test kümesine de uygulanır.  
5. **Eğitim/Test Ayrımı**  
   - `%80` eğitim  
   - `%20` test  
   - Sabit random seed ile tekrarlanabilir karıştırma.

---

## Model Mimarisi

Küçük bir tam bağlı (fully connected) sinir ağı:

1. **Girdi Katmanı**  
   - Boyut: 8 (8 öznitelik: `age`, `sex`, `bmi`, `children`, `smoker`, `region_northeast`, `region_northwest`, `region_southeast`, `region_southwest`)
2. **Gizli Katman**  
   - 16 nöron  
   - Aktivasyon: **ReLU**, \( 	ext{ReLU}(z) = \max(0, z) \)
3. **Çıktı Katmanı**  
   - 1 nöron (sigorta ücretini tahmin et)  
   - Aktivasyon: **Lineer** (regresyon için)

### Ağırlık Başlatma (Initialization)

- `W1` ve `W2` matrisleri: Normal dağılım (\(\mu=0\), \(\sigma=0.01\))  
- `b1`, `b2`: Sıfır vektörleri  

---

## Eğitim Süreci

- **Kayıp Fonksiyonu**: Mean Squared Error (MSE)  
  \[
  	ext{MSE} = rac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2
  \]
- **Optimizasyon**: Tam Küme Gradyan İnişi (Batch Gradient Descent)  
  - **Öğrenme Hızı** (\(lpha\)): 0.01  
  - **Epoch Sayısı**: 1.000  
  - **Batch Boyutu**: Tüm eğitim seti (batch size = m)

Her epoch:
1. **İleri Yayılım (Forward Pass)**  
   - Gizli katman:  
     \[
     Z^{[1]} = X W^{[1]} + b^{[1]},\quad A^{[1]} = 	ext{ReLU}(Z^{[1]})
     \]
   - Çıktı:  
     \[
     Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]},\quad \hat{y} = Z^{[2]}
     \]
2. **Kayıp Hesabı**  
   \[
   \mathcal{L} = rac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2
   \]
3. **Geri Yayılım (Backward Pass)**  
   - \(rac{\partial \mathcal{L}}{\partial Z^{[2]}}\), \(rac{\partial \mathcal{L}}{\partial W^{[2]}}\), \(rac{\partial \mathcal{L}}{\partial b^{[2]}}\) hesaplanır.  
   - ReLU’ya göre geriye doğru katman katman gradyanlar bulunur (\(W^{[1]}\), \(b^{[1]}\) için).
4. **Parametre Güncelleme**  
   \[
   W^{[i]} \leftarrow W^{[i]} - lpha \,rac{\partial \mathcal{L}}{\partial W^{[i]}}, \quad b^{[i]} \leftarrow b^{[i]} - lpha \,rac{\partial \mathcal{L}}{\partial b^{[i]}}
   \]

Eğitim sırasında her 100 epoch’ta bir şu çıktı yazdırılır:
```
Epoch 0, Loss: 0.4922
Epoch 100, Loss: 0.4913
Epoch 200, Loss: 0.4882
...
Epoch 900, Loss: 0.1243
```

Son olarak, hem eğitim hem test kümesi için **normalizasyonlu** MSE değerleri gösterilir.

---

## Sonuçlar

### 1. Korelasyon Matrisi

Aşağıdaki ısı haritası, sayısal ve tek-sıcak kodlanmış öznitelikler arasındaki Pearson korelasyon katsayılarını gösterir.  
- `smoker` ile `charges` arasındaki korelasyon ~0.79 ile çok yüksek.  
- `bmi` ve `charges` arasında orta düzeyde pozitif korelasyon (~0.20).

![Öznitelik Korelasyon Matrisi](plots/correlation_matrix.png)

### 2. Kayıp vs. Epoch

- 1.000 epoch boyunca öğrenme hızı = 0.01 ile eğitim yapıldı.  
- Başlangıçta (Epoch 0) kayıp ~0.4922.  
- Epoch 1.000 sonunda eğitim MSE ~0.12 (normalize edilmiş ölçekte).

![Epoch vs Kayıp (MSE)](plots/loss_curve.png)

#### Eğitim Kaybı Çıktıları

```
Epoch 0, Loss: 0.4922  
Epoch 100, Loss: 0.4913  
Epoch 200, Loss: 0.4882  
Epoch 300, Loss: 0.4766  
Epoch 400, Loss: 0.4366  
Epoch 500, Loss: 0.3367  
Epoch 600, Loss: 0.2193  
Epoch 700, Loss: 0.1593  
Epoch 800, Loss: 0.1352  
Epoch 900, Loss: 0.1243  

Final Training MSE (normalize edilmiş): 0.2381  
Final Test MSE (normalize edilmiş):     0.2196  
```

> **Not:** MSE değerleri, hedef (`charges`) ve öznitelikler 0–1 aralığına ölçeklendiği için normalize formatta verilmiştir.

### 3. Gerçek vs Tahmin (Test Kümesi)

Test kümesi için gerçek sigorta ücretleri (x ekseni) ile modelin tahminleri (y ekseni) karşılaştırıldı. Kırmızı kesikli çizgi \(y=x\) ideal tahmin hattını gösterir. Noktalar bu çizgiye ne kadar yakınsa, o kadar iyi tahmin demektir.

![Gerçek vs Tahmin (Test Kümesi)](plots/actual_vs_predicted.png)

#### İlk 10 Test Örneği Tahminleri

| İndeks | Gerçek Ücret | Tahmin Edilen Ücret |
|:------:|------------:|--------------------:|
| 0      | 9095.06825  | 8083.431183         |
| 1      | 5272.17580  | 7231.679608         |
| 2      | 29330.98315 | 35380.288919        |
| 3      | 9301.89355  | 8210.185429         |
| 4      | 33750.29180 | 27026.498442        |
| 5      | 4536.25900  | 11652.276124        |
| 6      | 2117.33885  | 5900.253956         |
| 7      | 14210.53595 | 17129.200207        |
| 8      | 3732.62510  | 6386.994884         |
| 9      | 10264.44210 | 8877.911203         |

---

## Çalıştırma Talimatları

1. **Depoyu klonlayın**  
   ```bash
   git clone https://github.com/kullanici_adiniz/sigorta‐nn‐scratch.git
   cd sigorta‐nn‐scratch
   ```
2. **Gerekli paketleri yükleyin**  
   ```bash
   pip install -r requirements.txt
   ```
3. **insurance.csv dosyasının kök dizinde olduğundan emin olun**  

4. **Python betiğini çalıştırın**  
   ```bash
   python ForwardAndBackwardPropagation.py
   ```
   - Her 100 epoch’ta bir eğitim kaybı konsola yazdırılır.  
   - Sonuçta şu çıktılar ve grafikler üretilir:  
     1. `plots/correlation_matrix.png`  
     2. `plots/loss_curve.png`  
     3. `plots/actual_vs_predicted.png`

5. **`plots/` klasörünü kontrol edin**  
   Oluşturulan grafikler bu klasörde saklanır. İstediğiniz zaman README’a ekleyebilir veya projenizde kullanabilirsiniz.

---

## Sonuç ve Değerlendirme

- **16 nöronlu gizli katman** ve **1.000 epoch** ile öğrenme hızı = 0.01 kullanılarak normalize edilmiş test MSE ≈ 0.22 elde edildi.  
- `smoker` özniteliği, sigorta ücretini tahmin etmede en güçlü tek başına etkendir (korelasyon ~0.79).  
- Model, yüksek ücretli uç değerlerde (özellikle sigara içenler ve yüksek BMI’ye sahipler) bazen hatalı tahmin yapsa da genel trendi yakalayabiliyor.  
- Sinir ağı adımları (ileri yayılım, geri yayılım, gradyan iniş) sıfırdan uygulandığından, bu çalışma derin öğrenmeye giriş için temel bir örnek niteliğindedir.

Dilediğiniz gibi hiperparametrelerle (gizli katman sayısı, öğrenme hızı, epoch sayısı vb.) oynamaktan ve ek düzenlemeler (L2 düzenleme, mini-batch öğrenme) eklemekten çekinmeyin.

---

### Teşekkür

- Kaggle’daki “Medical Cost Personal Datasets” veri seti.  
- NumPy ve Pandas dökümantasyonları.  
- İleri ve geri yayılım matematiksel kaynakları.  
