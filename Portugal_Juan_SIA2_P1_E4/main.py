# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:46:22 2024

@author: emman_pup4rlf
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('irisbin.csv', header=None)
X = data.iloc[:, :-3].values  
y = data.iloc[:, -3:].values  


species_names = ['Setosa', 'Versicolor', 'Virginica']
species_colors = ['red', 'green', 'blue']

for i in range(3):
    species_data = X[y[:, i] == 1]
    plt.scatter(species_data[:, 0], species_data[:, 1], label=species_names[i], c=species_colors[i])

plt.xlabel('1')
plt.ylabel('2')
plt.legend()
plt.title('Distribución')
plt.show()


k_out_accuracy = []
loo_accuracy = []


mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)


k_out = KFold(n_splits=5) 
loo = LeaveOneOut()

for train_index, test_index in k_out.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    k_out_accuracy.append(accuracy)

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loo_accuracy.append(accuracy)


k_out_error = 1 - np.mean(k_out_accuracy)
loo_error = 1 - np.mean(loo_accuracy)

k_out_avg_accuracy = np.mean(k_out_accuracy)
loo_avg_accuracy = np.mean(loo_accuracy)

k_out_std_deviation = np.std(k_out_accuracy)
loo_std_deviation = np.std(loo_accuracy)


print("Leave-k-Out - Error Esperado:", k_out_error)
print("Leave-One-Out - Error Esperado:", loo_error)
print("Leave-k-Out - Promedio de Exactitud:", k_out_avg_accuracy)
print("Leave-One-Out - Promedio de Exactitud:", loo_avg_accuracy)
print("Leave-k-Out - Desviación Estándar de Exactitud:", k_out_std_deviation)
print("Leave-One-Out - Desviación Estándar de Exactitud:", loo_std_deviation)

labels = ['Leave-k-Out', 'Leave-One-Out']
errors = [k_out_error, loo_error]
avg_accuracies = [k_out_avg_accuracy, loo_avg_accuracy]
std_deviations = [k_out_std_deviation, loo_std_deviation]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, errors, width, label='Error Esperado')
rects2 = ax.bar(x, avg_accuracies, width, label='Promedio Exactitud', bottom=errors)
rects3 = ax.bar(x, std_deviations, width, label='Desviación Estándar', bottom=np.array(errors) + np.array(avg_accuracies))

ax.set_xlabel('Validación')
ax.set_title('Resultados')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
