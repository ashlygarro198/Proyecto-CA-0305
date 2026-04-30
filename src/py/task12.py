#Archivo para trabajar en cada sección del módulo
import numpy as np

def one_hot_encode(indices: np.ndarray, num_classes: int) -> np.ndarray:
    
    # utilizamos la notacion de arrays de numpy
    indices = np.array(indices, dtype=np.int64)
    N = indices.shape[0]

    # array vacío
    if N == 0:
        return np.zeros((0, num_classes), dtype=np.float64)

    # caso indices invalidos
    if np.any(indices < 0) or np.any(indices >= num_classes):
        return np.zeros((N, num_classes), dtype=np.float64)

    # matriz de ceros
    resultado = np.zeros((N, num_classes), dtype=np.float64)

    # utilizar sugerencia
    resultado[np.arange(N), indices] = 1.0

    return resultado

# Ejemplo 1
indices = [0, 2, 1]     
num_classes = 3          
print(one_hot_encode(indices, num_classes))
