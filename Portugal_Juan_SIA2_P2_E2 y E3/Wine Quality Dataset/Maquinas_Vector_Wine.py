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


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    true_negatives = np.sum((y_pred == -1) & (y_true == -1))
    false_positives = np.sum((y_pred == 1) & (y_true == -1))
    false_negatives = np.sum((y_pred == -1) & (y_true == 1))
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
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

model = SVM()

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
