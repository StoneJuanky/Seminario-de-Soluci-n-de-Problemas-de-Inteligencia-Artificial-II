import numpy as np
from sklearn.model_selection import train_test_split

def cargar_datos(archivo):
    datos = np.genfromtxt(archivo, delimiter=',')
    entradas = datos[:, :-1]
    salidas = datos[:, -1]
    return entradas, salidas

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
    entradas, salidas = cargar_datos("spheres2d50.csv")
    num_particiones = 5
    porcentaje_entrenamiento = 0.8
    particiones = generar_particiones(entradas, salidas, num_particiones, porcentaje_entrenamiento)
    for i, particion in enumerate(particiones):
        print(f"Partición {i + 1}:")
        print(f"Patrones de entrenamiento: {len(particion['entradas_entrenamiento'])}")
        print(f"Patrones de prueba: {len(particion['entradas_prueba'])}")
        print("-----------")
