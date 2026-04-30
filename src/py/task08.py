#Archivo para trabajar en cada sección del módulo


import numpy as np

# Se define la funcion

def einsum_ops(A, B):
    
    # 1. Transpuesta de A
    # "ij->ji" intercambia los índices
    transpose = np.einsum("ij->ji", A)

    # 2. Suma total de todos los elementos
    # "ij->" significa que se eliminan todos los índices para obtener un escalar
    total_sum = np.einsum("ij->", A)

    # 3. Suma por filas
    # "ij->i" mantiene el índice i (filas) y suma sobre j (columnas)
    row_sum = np.einsum("ij->i", A)

    # 4. Suma por columnas
    # "ij->j" mantiene el índice j (columnas) y suma sobre i (filas)
    col_sum = np.einsum("ij->j", A)

    # 5. Multiplicación matricial
    # "ik,kj->ij": A tiene los índices (i,k) y B tiene índices (k,j). Se suma sobre k y el resultado tiene índices (i,j)
    matmul = np.einsum("ik,kj->ij", A, B)

    # Se devuelven todos los resultados en un diccionario
    return {
        "transpose": transpose,
        "sum": total_sum,
        "row_sum": row_sum,
        "col_sum": col_sum,
        "matmul": matmul
    }



A = np.array([[1, 2],
              [3, 4]])   

B = np.array([[5, 6],
              [7, 8]])  

print(einsum_ops(A, B))




