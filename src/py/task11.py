#Archivo para trabajar en cada sección del módulo
import numpy as np
from typing import Dict

def grad_matmul(grad_C: np.ndarray, A: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes gradients for C = AB.
    
    Args:
        grad_C: Upstream gradient dL/dC (M, N)
        A: Input A (M, K)
        B: Input B (K, N)
        
    Returns:
        Dict with "grad_A" and "grad_B"
    """
    # dL/dA = grad_C @ B^T
    # (M, N) @ (N, K) -> (M, K)
    grad_A = grad_C @ B.T
    
    # dL/dB = A^T @ grad_C
    # (K, M) @ (M, N) -> (K, N)
    grad_B = A.T @ grad_C
    
    return {
        "grad_A": grad_A,
        "grad_B": grad_B
    }

#Verificación con el ejemplo
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
grad_C = np.array([[0.1, 0.2], [0.3, 0.4]])

grads = grad_matmul(grad_C, A, B)

print("grad_A:\n", grads["grad_A"])
print("grad_B:\n", grads["grad_B"])

