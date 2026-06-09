#Archivo para trabajar en cada sección del módulo
import numpy as np

def softmax(x):
    x = np.array(x)

    #Encontrar el "logit" máximo
    max_i = np.max(x, axis=1, keepdims=True)

    #Se substrae el maximo de cada fila para evitar el overflow
    x_shifted = (x - max_i)

    #Se hace la exponencial para cada logit
    exp_x = np.exp(x_shifted)

    #Se computa el denominador (la suma de cada exponencial a lo largo de cada fila)
    S = np.sum(exp_x, axis=1, keepdims=True)

    #Se calcula el softmax diviendo cada exponencial por la suma
    softmax = exp_x / S

    return np.round(softmax, 3)

x = [[2.0, 1.0, 0.1],    # Sample 0: class 0 has highest logit
     [0.5, 2.5, 0.3]] 

print(softmax(x))

