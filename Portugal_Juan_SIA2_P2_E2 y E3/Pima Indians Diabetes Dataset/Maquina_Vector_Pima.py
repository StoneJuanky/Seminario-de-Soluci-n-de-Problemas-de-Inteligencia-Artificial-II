# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def rbf_kernel(X1, X2, gamma):
    pairwise_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * pairwise_dists)


def train_svm(X_train, y_train, C, gamma, max_iter):
    m, n = X_train.shape
    alpha = np.zeros(m)
    b = 0
    kernel = rbf_kernel(X_train, X_train, gamma)
    
    for _ in range(max_iter):
        for i in range(m):
            Ei = np.dot(alpha * y_train, kernel[i]) + b - y_train[i]
            if (y_train[i] * Ei < -0.001 and alpha[i] < C) or (y_train[i] * Ei > 0.001 and alpha[i] > 0):
                j = np.random.choice(list(range(i)) + list(range(i+1, m)))
                Ej = np.dot(alpha * y_train, kernel[j]) + b - y_train[j]
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]
                if y_train[i] != y_train[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                if L == H:
                    continue
                eta = 2 * kernel[i, j] - kernel[i, i] - kernel[j, j]
                if eta >= 0:
                    continue
                alpha[j] = alpha[j] - (y_train[j] * (Ei - Ej)) / eta
                alpha[j] = min(H, max(L, alpha[j]))
                if abs(alpha[j] - alpha_j_old) < 0.00001:
                    continue
                alpha[i] = alpha[i] + y_train[i] * y_train[j] * (alpha_j_old - alpha[j])
                b1 = b - Ei - y_train[i] * (alpha[i] - alpha_i_old) * kernel[i, i] - y_train[j] * (alpha[j] - alpha_j_old) * kernel[i, j]
                b2 = b - Ej - y_train[i] * (alpha[i] - alpha_i_old) * kernel[i, j] - y_train[j] * (alpha[j] - alpha_j_old) * kernel[j, j]
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
    return alpha, b


def predict_svm(X_test, X_train, y_train, alpha, b, gamma):
    kernel = rbf_kernel(X_train, X_test, gamma)
    predictions = np.dot((alpha * y_train).T, kernel) + b
    return np.sign(predictions).astype(int)


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=column_names)


X = data.drop('Outcome', axis=1)
y = data['Outcome']


X_normalized = normalize_data(X)


split_ratio = 0.8
split_index = int(len(data) * split_ratio)
X_train, X_test = X_normalized[:split_index], X_normalized[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


C = 1
gamma = 0.1
max_iter = 100
alpha, b = train_svm(X_train.values, y_train.values, C, gamma, max_iter)


y_pred = predict_svm(X_test.values, X_train.values, y_train.values, alpha, b, gamma)


accuracy = np.mean(y_pred == y_test)


true_positives = np.sum((y_pred == 1) & (y_test == 1))
false_positives = np.sum((y_pred == 1) & (y_test == 0))
false_negatives = np.sum((y_pred == 0) & (y_test == 1))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1_score:.2f}')
