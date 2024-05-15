import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def cargar_datos(archivo):
    datos = np.genfromtxt(archivo, delimiter=',')
    entradas = datos[:, :-1]
    salidas = datos[:, -1]
    return entradas, salidas


def entrenar_evaluar_perceptron(particion, tasa_aprendizaje, max_epocas, criterio_convergencia):
    entradas_entrenamiento = particion['entradas_entrenamiento']
    salidas_entrenamiento = particion['salidas_entrenamiento']
    entradas_prueba = particion['entradas_prueba']
    salidas_prueba = particion['salidas_prueba']   
    num_entradas = entradas_entrenamiento.shape[1]
    num_patrones = entradas_entrenamiento.shape[0]
    pesos = np.random.rand(num_entradas)
    bias = np.random.rand()
    
    epoca = 0
    convergencia = False
    
    while epoca < max_epocas and not convergencia:
        convergencia = True
        for i in range(num_patrones):
            entrada = entradas_entrenamiento[i]
            salida_deseada = salidas_entrenamiento[i]
            salida_obtenida = np.dot(pesos, entrada) + bias
            error = salida_deseada - salida_obtenida
            
            if abs(error) > criterio_convergencia:
                convergencia = False
                pesos += tasa_aprendizaje * error * entrada
                bias += tasa_aprendizaje * error
        
        epoca += 1
    salidas_predichas = np.sign(np.dot(entradas_prueba, pesos) + bias)
    accuracy = accuracy_score(salidas_prueba, salidas_predichas)
    
    return accuracy

def generar_particiones(entradas, salidas, num_particiones, porcentaje_entrenamiento):
    particiones = []
    for _ in range(num_particiones):
        entradas_entrenamiento, entradas_prueba, salidas_entrenamiento, salidas_prueba = train_test_split(
            entradas, salidas, test_size=(1 - porcentaje_entrenamiento), random_state=None)
        
        particion = {
            "entradas_entrenamiento": entradas_entrenamiento,
            "entradas_prueba": entradas_prueba,
            "salidas_entrenamiento": salidas_entrenamiento,
            "salidas_prueba": salidas_prueba
        }
        
        particiones.append(particion)
    
    return particiones


if __name__ == "__main__":
    entradas, salidas = cargar_datos("spheres2d70.csv")
    tasa_aprendizaje = 0.1
    max_epocas = 1000
    criterio_convergencia = 0.01
    num_particiones = 10
    porcentaje_entrenamiento = 0.8
    particiones = generar_particiones(entradas, salidas, num_particiones, porcentaje_entrenamiento)
    accuracies = []
    for i, particion in enumerate(particiones):
        accuracy = entrenar_evaluar_perceptron(particion, tasa_aprendizaje, max_epocas, criterio_convergencia)
        accuracies.append(accuracy)
        print(f"Partición {i + 1} - Accuracy: {accuracy}")
    promedio_accuracy = np.mean(accuracies)
    print(f"Promedio de accuracies: {promedio_accuracy}")
