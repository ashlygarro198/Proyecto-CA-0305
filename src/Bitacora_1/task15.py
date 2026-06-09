#Archivo para trabajar en cada sección del módulo

#Ejercicio 15


# %%
import numpy as np
# %%

# %%
def log_sum_exp(x):
    # Máximo por fila 
    maximo_fila = np.max(x, axis=1, keepdims=True)

    # Normalización de los exponentes restando el maximo
    x_ajustado = x - maximo_fila

    # Exponencial post ajuste
    exp_ajustados = np.exp(x_ajustado)

    # Suma por fila
    total_exp = np.sum(exp_ajustados, axis=1)

    # Logaritmo de la suma
    log_suma_exp = np.log(total_exp)

    # Reincorporar el máximo eliminado
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