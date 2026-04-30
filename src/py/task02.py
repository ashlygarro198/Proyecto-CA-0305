#Archivo para trabajar en cada sección del módulo
import numpy as np

def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using 3 nested loops.
    """
    #Dimensiones de la matriz
    M, K = A.shape
    K2, N = B.shape
    #Verificar que las columnas de A y las filas de B concuerde
    assert K == K2
    
    #Crea la matriz c llena de 0
    C = np.zeros((M, N))
    
    #Creamos ciclos for para que recorran tanto filas como columnas de la matriz 
    for i in range(M):
        for j in range(N):
            #definimos la suma que va a ser la entrada de cada ij de la matriz
            suma = 0
            for k in range(K):
                suma += A[i,k]*B[k,j]
            
            C[i,j]= suma 
    
    return C

def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Computes matrix product C = AB using vectorized operations.
    """
    return A @ B

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])
matriz = matmul_naive(A, B)
print(matriz)
matrizA=matmul_vectorized(A,B)
print(matrizA)