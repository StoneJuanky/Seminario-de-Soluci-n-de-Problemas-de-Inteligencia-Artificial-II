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


def gaussian_probability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), n_features))
        self.std = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx] = X_c.mean(axis=0)
            self.std[idx] = X_c.std(axis=0)
            self.priors[idx] = len(X_c) / n_samples
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(gaussian_probability(x, self.mean[idx], self.std[idx])))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
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

model = NaiveBayes()

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

# Calcula el F1 Score
f1 = f1_score(precision, sensitivity)
print(f'F1 Score: {f1:.2f}')
