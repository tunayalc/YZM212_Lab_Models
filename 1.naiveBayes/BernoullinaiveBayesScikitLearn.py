import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB

# Veriyi içe aktar
file_path = "Mushroom_Dataset/one_hot_encoded_mushroom_excel.xlsx"
df = pd.read_excel(file_path)

# Etiketleri oluştur (0_0 sütunu 1 ise 0, aksi halde 1)
y = df[['0_0', '0_1']].apply(lambda row: 0 if row['0_0'] == 1 else 1, axis=1)

# Özellikleri belirle (0_0 ve 0_1 sütunlarını çıkar)
X = df.drop(['0_0', '0_1'], axis=1)

# Veriyi eğitim ve test setlerine ayır (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scikit-learn BernoulliNB modelini oluştur ve eğit
start_time = time.time()
model_sk = BernoulliNB()
model_sk.fit(X_train, y_train)
train_time_sk = time.time() - start_time

# Model ile test verisini tahmin et
start_time = time.time()
y_pred_sk = model_sk.predict(X_test)
test_time_sk = time.time() - start_time

# Modelin doğruluğunu hesapla
accuracy_sk = accuracy_score(y_test, y_pred_sk)
cm_sk = confusion_matrix(y_test, y_pred_sk)

# Sonuçları ekrana yazdır
print("Scikit-learn BernoulliNB Sonuçları ")
print(f"Eğitim süresi: {train_time_sk:.4f} saniye")
print(f"Tahmin süresi: {test_time_sk:.4f} saniye")
print(f"Doğruluk (Accuracy): {accuracy_sk:.4f}")
print("Karmaşıklık Matrisi:")
print(cm_sk)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_sk))

# Karmaşıklık matrisini görselleştir
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.show()

plot_confusion_matrix(cm_sk, "Scikit-learn BernoulliNB - Confusion Matrix")
