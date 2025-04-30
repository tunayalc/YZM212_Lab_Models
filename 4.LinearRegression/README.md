
# Linear Regression Karşılaştırması

Bu çalışmada, `insurance.csv` veri seti kullanılarak iki farklı lineer regresyon modeli eğitilmiştir. Amaç, sağlık sigortası ücretlerini etkileyen faktörleri (yaş, BMI, çocuk sayısı, sigara kullanımı vb.) dikkate alarak tahmin yapan doğrusal modeller oluşturmaktır.

## Kullanılan Veri Seti

Veri seti, bireylerin yaş, cinsiyet, vücut kitle indeksi (BMI), çocuk sayısı, sigara içme durumu ve yaşadıkları bölge gibi demografik özelliklerini içermektedir. Hedef değişken `charges`, bireyin yıllık sağlık sigortası masrafını temsil etmektedir. Kategorik veriler one-hot encoding yöntemiyle sayısal hale getirilmiştir.

## Uygulanan Modeller

1. **LSE (Least Squares Estimation)**: En küçük kareler yöntemiyle kapalı formülle çözüm yapılmıştır. NumPy kullanılarak manuel olarak uygulanmıştır.
2. **Scikit-learn LinearRegression**: Python’un scikit-learn kütüphanesindeki `LinearRegression()` sınıfı kullanılarak model otomatik olarak eğitilmiştir.

## Ortalama Kare Hata (Mean Squared Error) Karşılaştırması

| Model                 | MSE (Mean Squared Error)       |
|-----------------------|--------------------------------|
| LSE (Manuel Yöntem)   | 36,501,893.00                  |
| Scikit-learn          | 36,501,893.00                  |

## Yorum

- Her iki modelde de aynı MSE değeri elde edilmiştir.
- Bu sonuç, hem manuel LSE yönteminin hem de scikit-learn kütüphanesinin aynı doğrusal çözümü ürettiğini göstermektedir.
- LSE yöntemi temel lineer cebir işlemleriyle çözüm üretirken, Scikit-learn bu işlemleri soyutlayarak daha pratik bir kullanım sunar.
- Eğitim açısından, LSE yöntemi algoritmanın iç yapısını anlamak için değerlidir; scikit-learn ise uygulamada zaman kazandırır.

