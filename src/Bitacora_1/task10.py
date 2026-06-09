#Archivo para trabajar en cada sección del módulo
import numpy as np
from typing import Tuple

def grad_sum(grad_y: float, x_shape: Tuple[int]) -> np.ndarray:
    """
    Computes gradient of x given gradient of sum(x).
    
    Args:
        grad_y: Scalar gradient dL/dy
        x_shape: Shape of the input tensor x
        
    Returns:
        Gradient dL/dx of shape x_shape
    """
    #np.full crea un array con la forma x_shape y lo llena con grad_y
    return np.full(shape=x_shape, fill_value=grad_y)

# --- Verificación con el ejemplo ---
x_shape = (3,)
grad_y = 0.5
f=grad_sum(grad_y, x_shape)
print(f)