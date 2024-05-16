# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Función para dividir los datos en conjuntos de entrenamiento y prueba
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    num_samples = X.shape[0]
    test_indices = np.random.choice(num_samples, size=int(num_samples * test_size), replace=False)
    train_indices = np.setdiff1d(np.arange(num_samples), test_indices)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

# Función para calcular la distancia euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Función para implementar el algoritmo de K-Vecinos Cercanos
class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.argmax(np.bincount(k_nearest_labels))
        return most_common

# Función para calcular la precisión de la clasificación
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Función para calcular la matriz de confusión
def confusion_matrix(y_true, y_pred):
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

# Función para calcular Precision, Sensitivity y Specificity
def precision_recall_specificity(y_true, y_pred, confusion):
    precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1]) if (confusion[1, 1] + confusion[0, 1]) > 0 else 0
    sensitivity = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0]) if (confusion[1, 1] + confusion[1, 0]) > 0 else 0
    specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1]) if (confusion[0, 0] + confusion[0, 1]) > 0 else 0
    return precision, sensitivity, specificity

# Función para calcular el F1 Score
def f1_score(precision, sensitivity):
    return 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

# Cargar los datos
url = "https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt"
data = pd.read_csv(url, sep="\t")

# Divide los datos en características (X) y etiquetas (y)
X = data.drop('Number of claims', axis=1).values
y = data['Number of claims'].values

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea un modelo de K-Vecinos Cercanos
model = KNN(k=5)

# Ajusta el modelo a los datos de entrenamiento
model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcula la precisión de la clasificación
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calcula la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(confusion)

# Calcula Precision, Sensitivity y Specificity
precision, sensitivity, specificity = precision_recall_specificity(y_test, y_pred, confusion)
print(f'Precision: {precision:.2f}')
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')

# Calcula el F1 Score
f1 = f1_score(precision, sensitivity)
print(f'F1 Score: {f1:.2f}')
