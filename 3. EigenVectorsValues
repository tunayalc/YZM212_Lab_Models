# Makine Öğrenmesinde Matris Manipülasyonu ve Özdeğer–Özvektör Analizi  
> Matris operasyonları ve eigen-analiz yöntemlerinin makine öğrenmesindeki uygulamalarının incelenmesi

---

## 1. Temel Kavramlar

### 1.1 Matris Manipülasyonu  
Veri setleriyle ve model parametreleriyle çalışırken elimizdeki tüm sayıları satır–sütun düzeni (vektör/matris) hâline getiriyoruz.  
Bu matrisler üzerinde toplama, çarpma, transpoze, tersini alma gibi işlemler yapıyor; ayrıca SVD veya özdeğer–özvektör ayrıştırmasıyla “iç yapıyı” dekompoze ediyoruz.

### 1.2 Özdeğer (λ)  
Bir \(A\) matrisinin, belirli bir \(v \neq 0\) vektörünü yalnızca ölçeklendirerek (yönünü değiştirmeden)  
\[
A v = \lambda v
\]  
eşitliğini sağladığı skaler sayıdır.

### 1.3 Özvektör (v)  
Yukarıdaki eşitliği sağlayan, yani matris çarpımında sadece \(\lambda\) katsayısı kadar uzayı genişleten veya daraltan vektördür.

---

## 2. Makine Öğrenmesindeki Rolleri

### 2.1 Boyut İndirgeme  
- **PCA**: Verinin kovaryans matrisini eigendecomposition’a sokup en büyük varyansı taşıyan eksenleri (ana bileşenleri) seçer.  
- **Kernel PCA**: Doğrusal olarak ayrılamayan veriyi yüksek boyutlu bir alana geçirip orada PCA uygulayarak doğrusal olmayan ilişkileri yakalar.

### 2.2 Gürültü Azaltma ve Özellik Çıkarımı  
- **SVD**: Veri matrisini \(U\,\Sigma\,V^\top\) formuna ayırır; \(\Sigma\)’deki küçük tekil değerleri atarak gereksiz detayları (gürültüyü) elemeyi sağlar.

### 2.3 Graf Tabanlı Kümeleme  
- **Spektral Kümeleme**: Verileri ilişkililik matrisine, oradan da Laplace operatörüne dönüştürür; en küçük özdeğerlere karşılık gelen özvektörleri alıp yeni bir temsilde klasik kümeleme algoritmalarını uygular.

### 2.4 Derin Öğrenmede Hesap Verimliliği  
- Backpropagation ve ağırlık güncellemeleri, büyük boyutlu matris çarpma/toplamaları üzerinden ilerler; GPU hızlandırmasıyla bu işlemler pratik hale gelir.

### 2.5 Yüz Tanıma (Eigenfaces)  
- Eğitim görüntülerinden oluşturulan yüz piksel matrisinin kovaryansını PCA’ya sokar, elde edilen temel yüz bileşenlerini (“eigenface”) çıkarır.  
- Yeni bir yüz, bu bileşenler uzayında küçük boyutlu bir vektör olarak temsil edilir ve tanıma benzerlik ölçütleriyle yapılır.

---

## 3. Yöntemler ve Uygulama Alanları

| Yöntem               | Temel İşlem                                          | Nerede Kullanılır                      |
|----------------------|------------------------------------------------------|----------------------------------------|
| **PCA**              | Kovaryans matrisini ayrıştırmak                      | Boyut indirgeme, görselleştirme        |
| **Kernel PCA**       | Veriyi karmaşık bir alana taşıyıp ayrıştırmak        | Doğrusal olmayan veri yapıları         |
| **SVD**              | Matrisi temel parçalara (tekil değerlere) ayırmak    | Gürültü filtresi, low-rank approx.     |
| **Spektral Kümeleme**| Graf Laplace operatörünü ayrıştırmak                 | Karmaşık ağlarda küme bulma            |
| **Eigenfaces**       | Yüz kovaryans matrisine PCA uygulamak                | Yüz tanıma, kimlik doğrulama           |

---

## 4. Kaynakça

- https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html  
- https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html  
- https://scikit-learn.org/stable/modules/decomposition.html  
- https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering  
- https://en.wikipedia.org/wiki/Eigenface  


# NUMPY.LINALG.EIG FONKSİYONU DOKÜMANTASYONU

## 1. NE İŞE YARAR?
Bir kare matris \(A \in \mathbb{C}^{M \times M}\) için
\[
A\,v_i = \lambda_i\,v_i
\]
eşitliğini sağlayan
- **ÖZDEĞERLER** \(\lambda_i\)
- **SAĞ ÖZVEKTÖRLER** \(v_i\)

çiftlerini hesaplar.

## 2. FONKSİYON İMZASI
```python
eigenvalues, eigenvectors = np.linalg.eig(a)
```
- **Parametreler**
  - `a` : Şekil `(..., M, M)` olan kare dizi veya dizi yığını
- **Dönüş Değerleri**
  - `eigenvalues` : Şekil `(..., M)` — \(\lambda_1, …, \lambda_M\)
  - `eigenvectors` : Şekil `(..., M, M)` — her sütun, karşılık gelen birim uzunlukta bir sağ özvektör  
- **Hata**
  - Yakınsamama durumunda `LinAlgError` yükselir.

## 3. PYTHON SEVİYESİNDEKİ ADIMLAR
1. **GİRDİ HAZIRLIĞI**  
   - `a`, `_makearray(a)` ile mutlaka `ndarray`’e dönüştürülür; orijinal tip için bir sarmalayıcı (wrap) fonksiyonu saklanır.
2. **BOYUT/KARE KONTROLLERİ**  
   - `_assert_stacked_2d(a)` ve `_assert_stacked_square(a)` ile son iki boyutun kare olduğu onaylanır.
3. **TÜR VE SIGNATURE BELİRLEME**  
   - `_commonType(a)` gerçek/karmaşık durumuna göre uygun NumPy tipi seçer.  
   - Buna göre `‘d->d’` veya `‘D->D’` gibi bir gufunc imzası (signature) belirlenir.
4. **GENELLEŞTİRİLMİŞ UFUNC ÇAĞRISI**  
   ```python
   w, v = _umath_linalg.eig(a, signature=signature)
   ```
5. **SONUCU PAKETLEME**  
   - Dönen `w` (özdeğerler) ve `v` (özvektörler), uygun NumPy tiplere cast edilir.  
   - `EigResult(w, v)` adlı `namedtuple` ile `wrap(...)` üzerinden kullanıcıya geri gönderilir.

## 4. C/C++ VE FORTRAN (LAPACK) ENTEGRASYONU
1. **GUFUNC TANIMI**  
   - Descriptor: `"(m,m)->(m),(m,m)"`  
   - Tür başına sarmalayıcılar: `DOUBLE_eig`, `ZDOUBLE_eig`, vb.
2. **BELLEK DÜZENLEME**  
   - Girdi matrisleri `linearize_matrix` ile tek boyutlu belleğe aktarılır.  
   - Fortran rutinleri `dgeev` (gerçek) veya `zgeev` (karmaşık) çağrılır.  
   - Dönen veriler `delinearize_matrix` ile yeniden çok boyutlu forma dönüştürülür.
3. **HATA YAKALAMA**  
   - LAPACK yakınsamazsa dönen hata kodu, Python’da `LinAlgError` olarak yükseltilir.

## 5. ÖZET AKIŞ ŞEMASI
```
Kullanıcı çağrısı
      ↓
makearray, assert kontrolleri
      ↓
commonType → signature belirleme
      ↓
→ _umath_linalg.eig (gufunc)
      ↓
dgeev / zgeev (LAPACK)
      ↓
Sonuçları linearize/delinearize
      ↓
EigResult olarak sar, wrap ile geri dön
```

---

> **Not:** Bu yapı sayesinde NumPy, esnek bir Python arayüzü ile yüksek performanslı Fortran/LAPACK hesaplamalarını bir arada sunar.

#NumPy Linalg.eig ve Saf Python Özdeğer Hesaplaması: Karşılaştırmalı İnceleme

## PROJE AÇIKLAMASI
Bu repository, NumPy'nın hazır fonksiyonu (`eig`) kullanmadan kare matrisler için özdeğer hesaplamasını saf Python ile nasıl gerçekleştirdiğini gösterir. Çalışmayı referans alarak tekrardan uygular ve aynı matris üzerinde NumPy `eig` fonksiyonuyla sonuçları karşılaştırır.

## 1. FONKSİYONLAR
- `get_dimensions(matrix)`  
  Verilen “liste içinde liste” formatındaki matrisi alır, `[satır_sayısı, sütun_sayısı]` döner.

- `find_determinant(matrix, excluded=1)`  
  - 2×2 matrisler için doğrudan `ad - bc` formülü.  
  - Daha büyük boyutlarda ilk satır üzerinden özyinelemeli kofaktör açılımı.  
  - `excluded` parametresi işaret değişim katsayılarını taşır.

- `list_multiply(list1, list2)` ve `list_add(list1, list2, sub)`  
  Polinom katsayılarını sırasıyla konvolüsyon (çarpma) veya eleman eleman toplama/çıkarma işlemleriyle birleştirir.

- `identity_matrix(dimensions)`  
  Verilen boyutlarda köşegeni 1, geri kalan elemanları 0 olan birim matris oluşturur.

- `characteristic_equation(matrix)`  
  Matrise `A - λI` formunda işlem uygular, her öğeyi `[a_{ij}, -δ_{ij}]` çifti hâline getirerek “liste içinde liste” formatında döner.

- `determinant_equation(matrix)`  
  `characteristic_equation` tarafından üretilen listeyi alır ve karakteristik polinomun katsayı listesini oluşturur.

- `find_eigenvalues(matrix)`  
  1. `determinant_equation` ile karakteristik polinom katsayılarını elde eder.  
  2. `numpy.roots` ile bu polinomun köklerini (özdeğerleri) bulur ve bir NumPy dizisi olarak döner.

## 2. KARŞILAŞTIRMA SONUCU

Görüldüğü üzere NumPy ve saf Python kodu aynı özdeğer kümesini (`{3, 5, 7}`) döndürüyor; custom yöntem Laplace açılımı kullanırken NumPy QR tabanlı algoritmayla hız ve sayısal kararlılık sağlar.
