import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Veriyi içe aktar
file_path = r"C:\Users\ytuna\OneDrive\Masaüstü\Naive Bayes\Mushroom_Dataset\one_hot_encoded_mushroom_excel.xlsx"
df = pd.read_excel(file_path)

# Etiketleri oluştur (0_0 sütunu 1 ise 0, aksi halde 1)
y = df[['0_0', '0_1']].apply(lambda row: 0 if row['0_0'] == 1 else 1, axis=1)

# Özellikleri belirle (0_0 ve 0_1 sütunlarını çıkar)
X = df.drop(['0_0', '0_1'], axis=1)

# Veriyi eğitim ve test setlerine ayır (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Kendi Bernoulli Naive Bayes modelimizi oluşturuyoruz
class CustomBernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X = np.array(X)
        y = np.array(y)
        self.feature_prob_ = {}
        self.class_prior_ = {}

        # Her sınıf için olasılıkları hesapla
        for c in self.classes_:
            X_c = X[y == c]
            prob = (np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + 2*self.alpha)
            self.feature_prob_[c] = prob
            self.class_prior_[c] = X_c.shape[0] / X.shape[0]
    
    def predict(self, X):
        X = np.array(X)
        predictions = []

        # Her örnek için en yüksek olasılıklı sınıfı bul
        for sample in X:
            log_probs = {}
            for c in self.classes_:
                log_prob = np.log(self.class_prior_[c])
                prob = self.feature_prob_[c]
                log_prob += np.sum(sample * np.log(prob) + (1 - sample) * np.log(1 - prob))
                log_probs[c] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        
        return np.array(predictions)

# Modeli eğit
start_time = time.time()
model_custom = CustomBernoulliNB(alpha=1.0)
model_custom.fit(X_train, y_train)
train_time_custom = time.time() - start_time

# Model ile test verisini tahmin et
start_time = time.time()
y_pred_custom = model_custom.predict(X_test)
test_time_custom = time.time() - start_time

# Modelin doğruluğunu hesapla
accuracy_custom = accuracy_score(y_test, y_pred_custom)
cm_custom = confusion_matrix(y_test, y_pred_custom)

# Sonuçları ekrana yazdır
print("Custom BernoulliNB Sonuçları ")
print(f"Eğitim süresi: {train_time_custom:.4f} saniye")
print(f"Tahmin süresi: {test_time_custom:.4f} saniye")
print(f"Doğruluk (Accuracy): {accuracy_custom:.4f}")
print("Karmaşıklık Matrisi:")
print(cm_custom)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_custom))

# Karmaşıklık matrisini görselleştir
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.show()

plot_confusion_matrix(cm_custom, "Custom BernoulliNB - Confusion Matrix")
