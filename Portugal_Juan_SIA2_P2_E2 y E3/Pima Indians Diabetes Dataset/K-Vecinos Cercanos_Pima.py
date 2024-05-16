# -*- coding: utf-8 -*-

import pandas as pd


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, names=column_names)


X = data.drop('Outcome', axis=1)
y = data['Outcome']


split_ratio = 0.8
split_index = int(len(data) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]


def euclidean_distance(x1, x2):
    return sum((xi - xj) ** 2 for xi, xj in zip(x1, x2)) ** 0.5


def find_neighbors(X_train, x_test, k):
    neighbors = []
    for i, x_train in enumerate(X_train):
        distance = euclidean_distance(x_train, x_test)
        neighbors.append((i, distance))
    neighbors.sort(key=lambda x: x[1])
    return neighbors[:k]


def predict_class(neighbors, y_train):
    votes = [y_train[i] for i, _ in neighbors]
    return max(set(votes), key=votes.count)


def evaluate_model(X_train, X_test, y_train, y_test, k):
    y_pred = []
    for x_test in X_test:
        neighbors = find_neighbors(X_train, x_test, k)
        predicted_class = predict_class(neighbors, y_train)
        y_pred.append(predicted_class)
    return y_pred


k = 5
y_pred = evaluate_model(X_train.values, X_test.values, y_train.values, y_test.values, k)


true_positives = sum((yt == 1) and (yp == 1) for yt, yp in zip(y_test, y_pred))
false_positives = sum((yt == 0) and (yp == 1) for yt, yp in zip(y_test, y_pred))
false_negatives = sum((yt == 1) and (yp == 0) for yt, yp in zip(y_test, y_pred))
true_negatives = sum((yt == 0) and (yp == 0) for yt, yp in zip(y_test, y_pred))

accuracy = (true_positives + true_negatives) / len(y_test)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)
f1_score = 2 * (precision * recall) / (precision + recall)

# Muestra las métricas
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Sensitivity (Recall): {recall:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1 Score: {f1_score:.2f}')
print('Confusion Matrix:')
print(f'True Positives: {true_positives}, False Positives: {false_positives}')
print(f'False Negatives: {false_negatives}, True Negatives: {true_negatives}')
