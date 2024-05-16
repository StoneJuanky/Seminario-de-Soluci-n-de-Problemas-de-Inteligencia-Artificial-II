import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import tkinter as tk
from tkinter import filedialog

def load_and_train_model():
    # Abrir un cuadro de diálogo para seleccionar el archivo CSV
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal de Tkinter
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not file_path:
        print("No file selected.")
        return

    # Cargar el dataset
    df = pd.read_csv(file_path)
    
    # Eliminar la columna animal_name
    df = df.drop(columns=['animal_name'])

    # Asumimos que la última columna es la etiqueta
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizar las características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Crear y entrenar el modelo SVM
    model = SVC(kernel='linear')  # Usando un kernel lineal como ejemplo
    model.fit(X_train, y_train)

    # Hacer predicciones y evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', pos_label=y_test.unique()[1])
    recall = recall_score(y_test, y_pred, average='weighted', pos_label=y_test.unique()[1])
    f1 = f1_score(y_test, y_pred, average='weighted', pos_label=y_test.unique()[1])
    
    # Obtener la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Calcular la especificidad si es una matriz de 2x2
    print(f'Modelo Support Vector Machines\n')
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        print(f'Specificity: {specificity:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Sensitivity (Recall): {recall:.2f}')
        print(f'Specificity: {specificity:.2f}')
        print(f'F1 Score: {f1:.2f}')
        
    
    print(f'Precisión del modelo: {accuracy:.2f}')
    print('Reporte de clasificación:')
    print(classification_report(y_test, y_pred))
    print('Matriz de confusión:')
    print(cm)

    # Guardar el modelo
    joblib.dump(model, 'svm_model.pkl')

    # Cargar el modelo (opcional)
    model = joblib.load('svm_model.pkl')

if __name__ == "__main__":
    load_and_train_model()

