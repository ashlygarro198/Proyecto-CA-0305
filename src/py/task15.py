#Archivo para trabajar en cada sección del módulo

#Ejerciio 15


# %%
import numpy as np
# %%

# %%
def log_sum_exp(x):
    maximo_fila = np.max(x, axis=1, keepdims=True)
    x_ajustado = x - maximo_fila
    exp_ajustados = np.exp(x_ajustado)
    total_exp = np.sum(exp_ajustados, axis=1)
    log_suma_exp = np.log(total_exp)
    resultado = log_suma_exp + maximo_fila.flatten()

    return resultado
# %%

# Ejemplo 1
x = np.array([[1000, 1001, 1002]])

print(log_sum_exp(x))
# %%

# %%

# Ejemplo 2
x = np.array([[2.0, 1.0, 0.1],
              [0.5, 2.5, 0.3]])

print(log_sum_exp(x))