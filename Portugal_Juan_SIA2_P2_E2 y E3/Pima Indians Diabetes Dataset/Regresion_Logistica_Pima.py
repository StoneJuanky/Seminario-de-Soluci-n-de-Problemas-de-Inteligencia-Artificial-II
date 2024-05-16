# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X_train, y_train, learning_rate, num_iterations):
    m, n = X_train.shape
    weights = np.zeros(n)
    bias = 0
    for i in range(num_iterations):
        z = np.dot(X_train, weights) + bias
        predictions = sigmoid(z)
        error = predictions - y_train
        gradient_weights = np.dot(X_train.T, error) / m
        gradient_bias = np.sum(error) / m
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
    return weights, bias


def predict_logistic_regression(X_test, weights, bias):
    z = np.dot(X_test, weights) + bias
    predictions = sigmoid(z)
    return predictions.round().astype(int)


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=column_names)


X = data.drop('Outcome', axis=1)
y = data['Outcome']


split_ratio = 0.8
split_index = int(len(data) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


learning_rate = 0.01
num_iterations = 1000
weights, bias = train_logistic_regression(X_train.values, y_train.values, learning_rate, num_iterations)


y_pred = predict_logistic_regression(X_test.values, weights, bias)


accuracy = np.mean(y_pred == y_test)
precision = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)
recall = np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)
specificity = np.sum((y_pred == 0) & (y_test == 0)) / np.sum(y_test == 0)
f1_score = 2 * (precision * recall) / (precision + recall)

# Muestra las métricas
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1 Score: {f1_score:.2f}')
