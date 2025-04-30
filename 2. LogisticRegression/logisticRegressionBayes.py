import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

columns = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

df = pd.read_csv(r'C:\Users\ytuna\OneDrive\Masaüstü\machine learning\LR\LogisticRegression\adult\adult.data',
                 names=columns,
                 sep=r",\s*",
                 engine="python")

df['income'] = df['income'].apply(lambda x: 1 if x.strip() == ">50K" else 0)

df = pd.get_dummies(df, columns=["workclass", "education", "marital-status",
                                 "occupation", "relationship", "race", "sex", "native-country"],
                    drop_first=True)

X = df.drop(columns=['income'])
y = df['income']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

class LogisticRegressionManual:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

manual_model = LogisticRegressionManual(lr=0.01, epochs=1000)

start_time = time.time()
manual_model.fit(X_train, y_train)
train_time_manual = time.time() - start_time

start_time = time.time()
y_pred_manual = manual_model.predict(X_test)
test_time_manual = time.time() - start_time

conf_mat_manual = confusion_matrix(y_test, y_pred_manual)
acc_manual = accuracy_score(y_test, y_pred_manual)

print("Manual Logistic Regression")
print("Eğitim Süresi: {:.4f} saniye".format(train_time_manual))
print("Test Süresi: {:.4f} saniye".format(test_time_manual))
print("Accuracy: {:.4f}".format(acc_manual))
print("Confusion Matrix:\n", conf_mat_manual)
print("Classification Report:\n", classification_report(y_test, y_pred_manual))

plt.figure(figsize=(6,5))
sns.heatmap(conf_mat_manual, annot=True, fmt='d', cmap='Reds')
plt.title('Manual Logistic Regression - Confusion Matrix')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()
