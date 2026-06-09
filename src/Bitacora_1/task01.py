#Archivo para trabajar en cada sección del módulo

import numpy as np
# Se crea la funcion de broadcasting 

def broadcasting_op(matriz, vector1, vector2):
    # Se reorganiza  vector2 a vector columna (N, 1) 
    vector2 = vector2.reshape(-1, 1)
    # Se suma el sesgo b a cada fila de X y se multiplica cada fila por su peso correspondiente
    resultado = (matriz + vector1) * vector2
    return resultado

m = np.array([[1, 2, 3],
              [4, 5, 6]])
v1 = np.array([10, 20, 30])
v2 = np.array([2, 3])

print(broadcasting_op(m , v1, v2))

