# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


data = pd.read_csv("concentlite.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
input_size = X_train.shape[1]
hidden_layers = [4, 4]
output_size = 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
np.random.seed(0)
weights = []
biases = []

layers = [input_size] + hidden_layers + [output_size]
for i in range(1, len(layers)):
    w = np.random.uniform(-1, 1, (layers[i-1], layers[i]))
    b = np.zeros((1, layers[i]))
    weights.append(w)
    biases.append(b)
learning_rate = 0.1
epochs = 10000
for epoch in range(epochs):
    layer_outputs = []
    layer_inputs = [X_train]
    
    for i in range(len(hidden_layers) + 1):
        z = np.dot(layer_inputs[i], weights[i]) + biases[i]
        a = sigmoid(z)
        layer_outputs.append(a)
        layer_inputs.append(a)
    error = y_train.reshape(-1, 1) - layer_outputs[-1]
    deltas = [error * sigmoid_derivative(layer_outputs[-1])]
    for i in range(len(hidden_layers), 0, -1):
        delta = deltas[-1].dot(weights[i].T) * sigmoid_derivative(layer_outputs[i])
        deltas.append(delta)
    deltas = deltas[::-1]
    for i in range(len(weights)):
        weights[i] += learning_rate * layer_inputs[i].T.dot(deltas[i])
        biases[i] += learning_rate * np.sum(deltas[i], axis=0)

layer_inputs = [X_test]
for i in range(len(hidden_layers) + 1):
    z = np.dot(layer_inputs[i], weights[i]) + biases[i]
    a = sigmoid(z)
    layer_inputs.append(a)

y_pred = layer_inputs[-1]
y_pred_class = np.round(y_pred)
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")


plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], label='Clase 0', c='b', marker='o')
plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], label='Clase 1', c='r', marker='s')
plt.title("Clasificación")
plt.xlabel("Mod 1")
plt.ylabel("Mod 2")
plt.legend()
plt.show()
