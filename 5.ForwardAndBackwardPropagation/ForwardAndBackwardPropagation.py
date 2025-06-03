import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/ytuna/OneDrive/Masaüstü/insurance.csv')
print(data.head())

data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data = pd.get_dummies(data, columns=['region'], drop_first=True)

plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Özelliklerin Korelasyon Matrisi")
plt.show()

X = data.drop('charges', axis=1).values
y = data['charges'].values.reshape(-1, 1)

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def activation(self, Z):
        if self.activation_name == 'relu':
            return np.maximum(0, Z)
        elif self.activation_name == 'tanh':
            return np.tanh(Z)
        else:
            raise ValueError("Unsupported activation.")

    def activation_derivative(self, Z):
        if self.activation_name == 'relu':
            return (Z > 0).astype(float)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(Z) ** 2
        else:
            raise ValueError("Unsupported activation.")

    def forward(self, X):
        activations = [X]
        pre_activations = []
        for i in range(len(self.weights) - 1):
            Z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(Z)
            A = self.activation(Z)
            activations.append(A)
        Z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(Z_out)
        activations.append(Z_out)
        return activations, pre_activations

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / (2 * m)
        return loss

    def backward(self, activations, pre_activations, y_true):
        m = y_true.shape[0]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)
        dA = activations[-1] - y_true
        grads_W[-1] = activations[-2].T @ dA / m
        grads_b[-1] = np.sum(dA, axis=0, keepdims=True) / m
        d_prev = dA
        for i in reversed(range(len(self.weights) - 1)):
            dZ = d_prev @ self.weights[i+1].T * self.activation_derivative(pre_activations[i])
            grads_W[i] = activations[i].T @ dZ / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m
            d_prev = dZ
        return grads_W, grads_b

    def update_parameters(self, grads_W, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_W[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            activations, pre_activations = self.forward(X)
            y_pred = activations[-1]
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            grads_W, grads_b = self.backward(activations, pre_activations, y)
            self.update_parameters(grads_W, grads_b)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return losses

    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]

input_size = X_train.shape[1]
hidden_neurons = 10
output_size = 1

nn = NeuralNetwork(layer_sizes=[input_size, hidden_neurons, output_size], activation='relu', learning_rate=0.01)
epochs = 1000
loss_history = nn.train(X_train, y_train, epochs=epochs)

plt.figure(figsize=(8, 5))
plt.plot(range(epochs), loss_history, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Loss vs. Epochs')
plt.legend()
plt.grid(True)
plt.show()

y_train_pred = nn.predict(X_train)
y_test_pred = nn.predict(X_test)
train_mse = np.mean((y_train_pred - y_train) ** 2)
test_mse = np.mean((y_test_pred - y_test) ** 2)
print(f"Final Train MSE: {train_mse:.4f}")
print(f"Final Test MSE: {test_mse:.4f}")

y_test_pred_original = scaler_y.inverse_transform(y_test_pred)
y_test_original = scaler_y.inverse_transform(y_test)

comparison_df = pd.DataFrame({
    "Actual": y_test_original.flatten(),
    "Predicted": y_test_pred_original.flatten()
}).head(10)
print(comparison_df)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test_original.flatten(), y=y_test_pred_original.flatten(), alpha=0.6)
min_val = min(y_test_original.min(), y_test_pred_original.min())
max_val = max(y_test_original.max(), y_test_pred_original.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Gerçek Charges")
plt.ylabel("Tahmin Edilen Charges")
plt.title("Gerçek vs. Tahmin (Test Kümesi)")
plt.grid(True)
plt.show()
