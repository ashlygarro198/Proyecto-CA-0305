#Archivo para trabajar en cada sección del módulo

import numpy as np


def batch_matmul(Q, K):
    
    # Convertimos los arreglos a unos que podamos trabajar con Numpy.
    Q = np.array(Q)
    K = np.array(K)  
    
    # Verificamos que tengan la misma forma antes de empezarlos a trabajar.
    if Q.shape != K.shape:
        raise ValueError("Q y K deben tener la misma forma (B,H,S,D)")
    
    # Usamos einsum (que hace operaciones tensoriales complejas (multiplicación, suma, transposición) 
    # en una sola línea) para calcular para cada (b,h) el produto punto entre el vector i y el vector j.
    
    # Referencia de donde investigamos más sobre einsum: https://medium.com/@whyamit404/what-is-numpy-einsum-and-why-use-it-47ea4492ddde 
    scores = np.einsum("bhid,bhjd->bhij", Q, K)    
    return scores


# Ejemplo
B, H, S, D = 2, 2, 3, 4 
Q = np.random.randn(B, H, S, D)
K = np.random.randn(B, H, S, D)
resultado = batch_matmul(Q, K)
print("Forma del resultado:", resultado.shape)

