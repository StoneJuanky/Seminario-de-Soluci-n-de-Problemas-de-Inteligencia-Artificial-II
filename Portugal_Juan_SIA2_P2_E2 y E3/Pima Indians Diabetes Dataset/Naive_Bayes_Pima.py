# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def normalize_data(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def gaussian_probability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent


def train_naive_bayes(X_train, y_train):
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    prior_probs = np.zeros(n_classes)
    class_means = np.zeros((n_classes, n_features))
    class_stds = np.zeros((n_classes, n_features))
    
    for c in range(n_classes):
        X_c = X_train[y_train == c]
        prior_probs[c] = len(X_c) / len(X_train)
        class_means[c] = np.mean(X_c, axis=0)
        class_stds[c] = np.std(X_c, axis=0)
    
    return prior_probs, class_means, class_stds


def predict_naive_bayes(X_test, prior_probs, class_means, class_stds):
    n_classes = len(prior_probs)
    n_samples = X_test.shape[0]
    predictions = np.zeros((n_samples, n_classes))
    
    for c in range(n_classes):
        class_prob = np.zeros(n_samples)
        for i in range(n_samples):
            likelihood = np.prod(gaussian_probability(X_test[i], class_means[c], class_stds[c]))
            class_prob[i] = likelihood * prior_probs[c]
        predictions[:, c] = class_prob
    
    return np.argmax(predictions, axis=1)


def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    return accuracy, precision, sensitivity, specificity, f1_score

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


prior_probs, class_means, class_stds = train_naive_bayes(X_train.values, y_train.values)


y_pred = predict_naive_bayes(X_test.values, prior_probs, class_means, class_stds)


accuracy, precision, sensitivity, specificity, f1_score = calculate_metrics(y_test.values, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Sensitivity (Recall): {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1 Score: {f1_score:.2f}')
