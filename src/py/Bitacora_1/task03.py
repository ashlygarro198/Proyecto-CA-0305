#Archivo para trabajar en cada sección del módulo


import numpy as np

#  Se crea la fucion que realiza operaciones elemento a elemento entre dos tensores del mismo tamaño
def elementwise_ops(a, b):

    # Se define constante pequeña para evitar división por cero
    eps = 1e-8

    # Se suma elemento a elemento
    add = a + b

    # Se multilica elemento a elemento 
    mul = a * b

    # Se crea una division segura: se suma eps al denominador para evitar división por cero
    div = a / (b + eps)

    # Se retornan los resultados en un diccionario
    return {
        "add": add,
        "mul": mul,
        "div": div
    }

a = np.array([1.0, 2.0])
b = np.array([0.0, 2.0])

print(elementwise_ops(a, b))