#Archivo para trabajar en cada sección del módulo

import numpy as np


def compute_norms(x):
    x = np.array(x)
    
    normas = {"l1": np.sum(np.abs(x), axis=1),   #La norma L1 es la suma del valor absoluto de cada valor
        "l2": np.sqrt(np.sum(x**2, axis=1))      #La L2 es la raiz cuadrado de la suma de cada valor al cuadrado
    }                                            #Esto se calcula a lo largo de cada fila para cada norma y retorna un diccionario con estas
    return normas

x = [[3, 4],      # First vector: (3, 4)
     [1, -1]]

print(compute_norms(x))


