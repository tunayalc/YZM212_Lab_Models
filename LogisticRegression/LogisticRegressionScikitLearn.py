import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

columns = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

df = pd.read_csv('adult.data' ,  names=columns, sep=",\s*", engine="python")

df['income'] = df['income'].apply(lambda x: 1 if x.strip() == ">50K" else 0)

df = pd.get_dummies(df, columns=["workclass", "education", "marital-status",
                                 "occupation", "relationship", "race", "sex", "native-country"],
                    drop_first=True)

X = df.drop(columns=['income'])
y = df['income']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

sk_model = LogisticRegression(solver='lbfgs', max_iter=1000)

start_time = time.time()
sk_model.fit(X_train, y_train)
train_time_sk = time.time() - start_time

start_time = time.time()
y_pred_sk = sk_model.predict(X_test)
test_time_sk = time.time() - start_time

conf_mat_sk = confusion_matrix(y_test, y_pred_sk)
acc_sk = accuracy_score(y_test, y_pred_sk)

print("Scikit-Learn Logistic Regression")
print("Eğitim Süresi: {:.4f} saniye".format(train_time_sk))
print("Test Süresi: {:.4f} saniye".format(test_time_sk))
print("Accuracy: {:.4f}".format(acc_sk))
print("Confusion Matrix:\n", conf_mat_sk)
print("Classification Report:\n", classification_report(y_test, y_pred_sk))

plt.figure(figsize=(6,5))
sns.heatmap(conf_mat_sk, annot=True, fmt='d', cmap='Blues')
plt.title('Scikit-Learn Logistic Regression - Confusion Matrix')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()
