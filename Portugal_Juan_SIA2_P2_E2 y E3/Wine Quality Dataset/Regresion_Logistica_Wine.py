# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    num_samples = X.shape[0]
    test_indices = np.random.choice(num_samples, size=int(num_samples * test_size), replace=False)
    train_indices = np.setdiff1d(np.arange(num_samples), test_indices)
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient
    
    def predict(self, X):
        return np.round(self.sigmoid(np.dot(X, self.theta))).astype(int)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[true_negatives, false_positives], [false_negatives, true_positives]])


def precision_recall_specificity(y_true, y_pred, confusion):
    precision = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1]) if (confusion[1, 1] + confusion[0, 1]) > 0 else 0
    sensitivity = confusion[1, 1] / (confusion[1, 1] + confusion[1, 0]) if (confusion[1, 1] + confusion[1, 0]) > 0 else 0
    specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1]) if (confusion[0, 0] + confusion[0, 1]) > 0 else 0
    return precision, sensitivity, specificity


def f1_score(precision, sensitivity):
    return 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
data = pd.read_csv(url, sep=';')  


X = data.drop('quality', axis=1)  
y = (data['quality'] > 6).astype(int)  
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
confusion = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(confusion)
precision, sensitivity, specificity = precision_recall_specificity(y_test, y_pred, confusion)
print(f'Precision: {precision:.2f}')
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
f1 = f1_score(precision, sensitivity)
print(f'F1 Score: {f1:.2f}')
