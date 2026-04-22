#Archivo para trabajar en cada sección del módulo

import numpy as np

def tensor_reductions(x, axis):
    x = np.array(x) 
    #Se crea un diccionario que computa cada operacion con respecto al axis indicado
    reductions = {"sum": np.sum(x, axis=axis),
            "mean": np.mean(x, axis=axis),
            "max": np.max(x, axis=axis),
            "argmax": np.argmax(x, axis=axis)
    }
    return reductions

x = [[1, 2, 3], 
     [4, 5, 6]]  # Shape (2, 3)
axis = 1

print(tensor_reductions(x, axis))

