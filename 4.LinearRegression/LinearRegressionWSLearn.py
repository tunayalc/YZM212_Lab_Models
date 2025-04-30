import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:\\Users\\ytuna\\OneDrive\\Masaüstü\\insurance.csv")
data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)
X = data.drop("charges", axis=1).values
y = data["charges"].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("Scikit-learn Model MSE:", mse)

plt.figure(figsize=(10, 5))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahminler")
plt.title("Scikit-learn Linear Regression: Gerçek vs Tahmin")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
