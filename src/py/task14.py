#Archivo para trabajar en cada sección del módulo
#Ejercicio 14 

import numpy as np

def cross_entropy_loss(probs, targets):

    # Número de ejemplos
    N = len(targets)

    # Probabilidad asignada a la clase correcta
    correct_probs = probs[np.arange(N), targets]

    # este es el epsilon para evitar conflictos numericos
    eps = 1e-9

    # Logaritmo de las probabilidades correctas
    log_probs = np.log(correct_probs + eps)

    # Promedio de la pérdida negativa
    loss = -np.mean(log_probs)

    return loss
# %%

# %%
# Ejemplo 1
probs = np.array([[0.1, 0.9],
                  [0.8, 0.2]])

targets = np.array([1, 0])

print(cross_entropy_loss(probs, targets))

# %%

# %%
# Ejemplo 2
probs = np.array([[0.9, 0.1],
                  [0.2, 0.8]])

targets = np.array([1, 0])

print(cross_entropy_loss(probs, targets))
# %%