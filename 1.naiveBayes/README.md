Naive Bayes Projesi

Bu projede, mantarların zehirli mi yoksa yenilebilir mi olduğunu tahmin etmek için Bernoulli Naive Bayes algoritması kullanılmıştır. Uygulama iki farklı şekilde gerçekleştirilmiştir. Birinci yöntem Scikit-learn kütüphanesi ile hazır BernoulliNB modeli kullanılarak yapılmıştır. İkinci yöntem ise Python ile elle yazılmış Bernoulli Naive Bayes algoritmasının uygulanmasıdır.

Veri seti olarak Mushroom Dataset (UCI ML Repository) kullanılmıştır. Toplam 8124 veri bulunmaktadır. Veri seti 22 kategorik özelliğe sahiptir. Hedef değişken mantarın yenilebilir veya zehirli olup olmadığını belirten class sütunudur. Veri setinde eksik veri bulunmamaktadır.

Ham veri doğrudan okunabilir bir formatta olmadığı için anlaşılabilir hale getirmek amacıyla excel formatına çevrilmiştir. Mushroom excel dosyası oluşturulmuş ve verilerin daha rahat incelenmesi sağlanmıştır. Daha sonra kod kalabalığını önlemek ve modelin çalışabilmesi için tüm kategorik değişkenler sayısal hale getirilmiştir. Bu işlem sonucunda one hot encoded mushroom excel dosyası oluşturulmuştur.

Bu veri seti teorik olarak Multinomial Naive Bayes için daha uygun görünse de pratikte frekans bazlı analiz gerektirmediği için Bernoulli Naive Bayes kullanmak daha mantıklıdır. Veriyi one-hot encoding ile ikili forma çevirerek Bernoulli modeliyle daha doğru tahminler elde edilmiştir.

İki modelin performansı karşılaştırılmıştır. Scikit-learn ile oluşturulan modelin doğruluk oranı yüzde 93.60 olarak hesaplanmıştır. Eğitilme süresi 0.0000 saniye, tahmin süresi 0.0000 saniyedir. Elle yazılmış Bernoulli Naive Bayes modeli de aynı doğruluk oranını vermiştir. Ancak eğitim süresi 0.0156 saniye, tahmin süresi ise 0.0312 saniye olarak ölçülmüştür. Bu sonuçlara göre, elle yazılmış modelin çalışma süresi daha uzundur.

İki modelin karmaşıklık matrisi aynıdır. Karmaşıklık matrisi şu şekildedir:

827 doğru tahmin edilen yenilebilir mantar, 16 yanlış tahmin edilen yenilebilir mantar.  
88 yanlış tahmin edilen zehirli mantar, 694 doğru tahmin edilen zehirli mantar.  

Sınıflandırma raporunda precision, recall ve f1-score değerleri her iki model için de benzer çıkmıştır. Yenilebilir mantarlar için precision değeri 0.90, recall değeri 0.98, f1-score değeri 0.94 olarak hesaplanmıştır. Zehirli mantarlar için precision değeri 0.98, recall değeri 0.89, f1-score değeri 0.93’tür. Genel doğruluk oranı 0.94 olarak ölçülmüştür.

Sonuç olarak, iki modelin doğruluk oranı aynıdır ancak Scikit-learn modeli optimizasyon açısından çok daha hızlıdır. Ham verinin excel formatına dönüştürülmesi, veri setinin daha kolay analiz edilmesini sağlamıştır. Ayrıca kod kalabalığını önlemek ve modelin çalışmasını kolaylaştırmak için one hot encoding yöntemi kullanılmıştır. Multinomial yerine Bernoulli Naive Bayes kullanımı, verinin varlık/yokluk bazlı olması nedeniyle daha doğru sonuçlar üretmiştir.

Kaynakça :

https://www.youtube.com/watch?v=TLInuAorxqE  
https://github.com/AssemblyAI-Community/Machine-Learning-From-Scratch/blob/main/06%20NaiveBayes/naive_bayes.py  
https://www.quora.com/What-is-the-best-way-to-use-continuous-variables-for-a-naive-bayes-classifier-Do-we-need-to-cluster-them-or-leave-for-self-learning-Pls-help  
https://stackoverflow.com/questions/14254203/mixing-categorial-and-continuous-data-in-naive-bayes-classifier-using-scikit-lea  
https://www.geeksforgeeks.org/gaussian-naive-bayes/  
https://www.geeksforgeeks.org/bernoulli-naive-bayes/  
https://www.geeksforgeeks.org/multinomial-naive-bayes/  
https://www.youtube.com/watch?v=JjY4NJyUV1I&list=PL3ED48mWmYxrAdWjQlOWzFNaM4gLgry5T&index=19  

Ahmet Tunahan Yalçın 22290665
