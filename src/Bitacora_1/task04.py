#Archivo para trabajar en cada sección del módulo

import numpy as np


def reshape_and_transpose(x, B, C, H, W):
    
    # Hay que verificar que el tamaño coincida, ya que si no coincide no podemos realizar las operaciones.
    if len(x) != B * C * H * W:
        raise ValueError("El tamaño del vector no coincide con B*C*H*W")
        
        
    # Ya verificado lo primero es convertir la lista a arreglo NumPy para así poder trabajar con ella.
    arr = np.array(x)
    
    # Luego utilizamos la función de numpy reshape que lo que va a hacer es ordenar el arreglo en una matriz (B, C, H, W).
    # Referencia de donde buscamos la función reshape para saber como utilizarla: https://interactivechaos.com/es/manual/tutorial-de-numpy/la-funcion-reshape 
    arr = arr.reshape(B, C, H, W)
    
    # Finalmente  con la función transpose vamos a transponer a (B, H, W, C).
    # Referencia para el uso de la función de numpy transpose: https://www.geeksforgeeks.org/python/python-numpy-numpy-transpose/ 
    arr = arr.transpose(0, 2, 3, 1)
    return arr

# Ejemplo

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

B, C, H, W = 1, 2, 3, 4

resultado = reshape_and_transpose(x, B, C, H, W)
print(resultado)
print("Forma final:", resultado.shape)
