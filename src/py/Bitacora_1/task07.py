#Archivo para trabajar en cada sección del módulo

import numpy as np
def vector_products(a, b):
    
    # Debemos convertir a arreglos NumPy para poder trabajarlos.
    a = np.array(a)
    b = np.array(b)
    
    # Verificar que tengan forma (N,3) para poder trabajar con ellos.
    if a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("Los vectores deben tener 3 componentes (x,y,z)")
    
    # Producto escalar, en este lo que ocupamos es ir sumando la multiplicación de las entradas. 
    dot = np.sum(a * b, axis=1) 
    
    
    # Producto vectorial, en este numpy tiene una función que nos ayuda a realizar el proceso.
    # Referencia de donde encontramos la función cross https://docs.vultr.com/python/third-party/numpy/cross 
    cross = np.cross(a, b)
    return {"dot": dot, "cross": cross}


#Ejemplo
a = [[1, 0, 0]]
b = [[0, 1, 0]]
resultado = vector_products(a, b)
print("Producto escalar:", resultado["dot"])
print("Producto Vectorial:", resultado["cross"])
