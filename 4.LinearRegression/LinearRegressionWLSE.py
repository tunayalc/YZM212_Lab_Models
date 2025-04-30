import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\ytuna\\OneDrive\\Masaüstü\\insurance.csv")
data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)
X = data.drop("charges", axis=1).astype(float).values
y = data["charges"].values.reshape(-1, 1)
X = np.hstack([np.ones((X.shape[0], 1)), X])

X_transpose = X.T
theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
y_pred = X @ theta
mse = np.mean((y - y_pred) ** 2)
print("LSE Model MSE:", mse)

plt.figure(figsize=(10, 5))
plt.scatter(y, y_pred, alpha=0.5)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahminler")
plt.title("LSE Linear Regression: Gerçek vs Tahmin")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
