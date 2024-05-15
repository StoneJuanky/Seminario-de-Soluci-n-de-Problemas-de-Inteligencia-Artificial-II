import numpy as np
import matplotlib.pyplot as plt

# Función para leer patrones de un archivo CSV
def leer_patrones(archivo):
    datos = np.genfromtxt(archivo, delimiter=',')
    entradas = datos[:, :-1]
    salidas = datos[:, -1]
    return entradas, salidas

# Función para entrenar el perceptrón
def entrenar_perceptron(entradas, salidas, tasa_aprendizaje, max_epocas, criterio_convergencia):
    num_entradas = entradas.shape[1]
    num_patrones = entradas.shape[0]
    
    # Inicialización de pesos y bias
    pesos = np.random.rand(num_entradas)
    bias = np.random.rand()
    
    epoca = 0
    convergencia = False
    
    while epoca < max_epocas and not convergencia:
        convergencia = True
        for i in range(num_patrones):
            entrada = entradas[i]
            salida_deseada = salidas[i]
            salida_obtenida = np.dot(pesos, entrada) + bias
            error = salida_deseada - salida_obtenida
            
            if abs(error) > criterio_convergencia:
                convergencia = False
                pesos += tasa_aprendizaje * error * entrada
                bias += tasa_aprendizaje * error
        
        epoca += 1
    
    return pesos, bias

# Función para probar el perceptrón entrenado
def probar_perceptron(entradas, pesos, bias):
    salida_obtenida = np.dot(entradas, pesos) + bias
    return np.sign(salida_obtenida)

# Función para graficar los patrones y la recta que los separa
def graficar_patrones_recta(entradas, salidas, pesos, bias):
    plt.figure(figsize=(8, 6))
    
    # Graficar patrones
    plt.scatter(entradas[:, 0], entradas[:, 1], c=salidas, cmap=plt.cm.Paired, marker='o', s=100)
    
    # Graficar recta de separación
    x_min, x_max = entradas[:, 0].min() - 1, entradas[:, 0].max() + 1
    y_min, y_max = entradas[:, 1].min() - 1, entradas[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = probar_perceptron(np.c_[xx.ravel(), yy.ravel()], pesos, bias)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, colors='k', linestyles=['-'], levels=[0])
    
    plt.title('Patrones y Recta de Separación')
    plt.xlabel('Entrada 1')
    plt.ylabel('Entrada 2')
    plt.show()

# Main
if __name__ == "__main__":
    # Lectura de patrones de entrenamiento
    entradas_entrenamiento, salidas_entrenamiento = leer_patrones("XOR_trn.csv")
    
    # Parámetros de entrenamiento
    tasa_aprendizaje = 0.1
    max_epocas = 1000
    criterio_convergencia = 0.01
    
    # Entrenar el perceptrón
    pesos_entrenados, bias_entrenado = entrenar_perceptron(
        entradas_entrenamiento, salidas_entrenamiento, tasa_aprendizaje, max_epocas, criterio_convergencia)
    
    # Lectura de patrones de prueba
    entradas_prueba, salidas_prueba = leer_patrones("XOR_tst.csv")
    
    # Probar el perceptrón entrenado en datos de prueba
    salidas_predichas = probar_perceptron(entradas_prueba, pesos_entrenados, bias_entrenado)
    
    # Mostrar resultados
    print("Salidas reales en datos de prueba:")
    print(salidas_prueba)
    print("Salidas predichas por el perceptrón:")
    print(salidas_predichas)
    
    # Graficar patrones y la recta que los separa
    graficar_patrones_recta(entradas_entrenamiento, salidas_entrenamiento, pesos_entrenados, bias_entrenado)
