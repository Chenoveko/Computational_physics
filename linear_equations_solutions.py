import numpy as np


# --------------------------------------- Metodo de Gauss----------------------------------------------------------------
def gaussian_elimination(A: 'Coeficients matrix',
                         B: 'Column vector') -> 'Solution of the system using gaussian elimination and matrix extendida triangular':
    """
    La idea del metodo es utilizar operaciones elementales de matrices que permitan transformar
    nuestro sistema de ecuaciones inicial en uno donde la matriz A sea triangular.
    """
    # Calculamos el número de filas
    N = len(B)
    # Create our augmented matrix
    AB = np.column_stack((A, B))
    # Eliminación gaussiana
    for m in range(N):
        # 1. dividimos por el elemento
        div = AB[m, m]
        AB[m, :] /= div
        # 2. Sustraemos las filas siguientes
        for i in range(m + 1, N):
            mult = AB[i, m]
            AB[i, :] -= mult * AB[m, :]
        # 3. Sustitución inversa
        x = np.zeros(N, float)
        for m in range(N - 1, -1, -1):
            x[m] = AB[m, N]
            for i in range(m + 1, N):
                x[m] -= AB[m, i] * x[i]
    return x, AB


def gaussian_elimination_pivot(A: 'Coeficients matrix',
                               B: 'Column vector') -> 'Solution of the system using gaussian elimination with pivot':
    """
    - La eliminación gaussiana requiere dividir en cada paso por el elemento aii, lo cual sólo es
    posible si el elemento es diferente de cero.
    - El metodo del pivote combina la eliminación gaussiana con la operación elemental 3 de las
    matrices: el intercambio de una fila por otro.
    - Esto garantiza que en ningún paso un elemento de la diagonal sea 0.
    """

    # Calculamos el número de filas
    N = len(B)
    # Creamos nuestra matriz ampliada
    AB = np.column_stack((A, B))
    # Eliminación gaussiana con metodo del pivote
    for m in range(N):
        # 1. Escogemos la fila i>=m cuyo elemento a_{im} este más alejado de 0
        pivot = m
        largest = abs(AB[m, m])
        for i in range(m + 1, N):
            if abs(AB[i, m] > largest):
                largest = AB[i, m]
                pivot = i
        # 2. Intercambiamos la fila pivot por la fila m.
        if pivot != m:
            for i in range(N + 1):
                AB[m, i], AB[pivot, i] = AB[pivot, i], AB[m, i]
        # 3. dividimos por el elemento
        div = AB[m, m]
        AB[m, :] /= div
        # 4. Sustraemos las filas siguientes
        for i in range(m + 1, N):
            mult = AB[i, m]
            AB[i, :] -= mult * AB[m, :]

    # 5. Sustitución inversa
    x = np.zeros(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = AB[m, N]
        for i in range(m + 1, N):
            x[m] -= AB[m, i] * x[i]
    return x, AB


# ---------------------------------------------Metodo LU-----------------------------------------------------------------

def LU_factorization(A: 'Matrix to factorize') -> 'Lower Upper factorization of the matrix A':
    """
    - La eliminación gaussiana con pivote es un metodo efectivo y rápido.
    - Sin embargo, en muchas ocasiones en física queremos resolver el sistema: A · X = V, para distintos valores de V.
    """
    N = len(A)
    # Inicializamos nuestras dos matrices.
    L = np.zeros([N, N], float)  # L tiene que ser una matriz triangular inferior.
    U = np.copy(A)  # U será la matriz A convertida en triangular superior
    # Creamos la matrix L
    for m in range(N):
        L[m:N, m] = U[m:N,
                    m]  # para cada iteracción (para cada fila), la columna m de L queda fijada por el valor que tiene U.
        # Convertimos ahora U en una matrix triangular superior.
        # Para ello usamos la eliminación guassiana.
        # 1. Dividimos la fila m por el elemento m,m
        div = U[m, m]
        U[m, :] /= div
        # 2. Sustraemos la fila m a las filas i>m multiplicadas por el elemento i,m
        for i in range(m + 1, N):
            mult = U[i, m]
            U[i, :] -= mult * U[m, :]
    return L, U


# ---------------------------LU method------------------------------

def LU_solution(A: 'Coeficients matrix', V: 'Column vector') -> 'Solution of the system using LU method':
    N = len(A)
    L, U = LU_factorization(A)
    # 1. Sustitución hacía adelante. Operando con L.
    y = np.empty(N, float)
    for m in range(N):
        y[m] = V[m]
        for i in range(m):
            y[m] -= L[m, i] * y[i]
        y[m] /= L[m, m]
    # 2. Sustitución hacía atrás. Operando con U.
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = y[m]
        for i in range(m + 1, N):
            x[m] -= U[m, i] * x[i]
    return x


# -----------------------------------------------Utilizando la librería numpy-------------------------------------------

def numpy_solver(A: 'Coeficients matrix', B: 'Column vector') -> 'Solution of the system using numpy method':
    return np.linalg.solve(A, B)


# ---------------------------------------------Métodos iterativos para ecuaciones lineales-------------------------------
"""
- La factorización LU conlleva 2/3 N^3 operaciones, lo que para sistemas de ecuaciones grandes puede ser muy costoso computacionalmente.
- Como alternativa, se suelen usar los métodos iterativos, que en general sólo son prácticos en 
  ciertas situaciones, por ejemplo con matrices con muchos elementos no diagonales nulos.
"""


def jacobi_method(A, v, eps):
    # eps es el error de convergencia
    N = len(v)
    # Creamos nuestra descompsición D,L,U
    D = np.diag(A)  # diag crea un vector con los elementos de la diagonal
    LU = A - np.diagflat(D)  # diagflat crea una matriz diagonal cuyo argumento es un vector.
    x0 = np.zeros(N, float)  # creamos la primera estimación de X
    err = 1e6
    it = 0
    while err > eps:
        x = (v - np.dot(LU, x0)) / D  # iteramos con X
        err = max(abs(x - x0))
        x0 = np.copy(x)
        it += 1
    return x, it


def gauss_seidel_method(A, v, eps):
    # eps es el error de convergencia
    N = len(v)
    # Creamos nuestra descomposición D,L,U
    DL = np.tril(A)  # tril crea un matriz triangular inferior
    U = A - DL  # creamos la matrix U
    x0 = np.zeros(N, float)  # creamos la primera estimación de x
    # inicializamos x
    err = 1e6
    it = 0
    while err > eps:
        x = (v - np.dot(U, x0))
        # Sustitución hacía adelante
        for m in range(N):
            for i in range(m):
                x[m] -= DL[m, i] * x[i]
            x[m] /= DL[m, m]
        err = max(abs(x - x0))
        x0 = np.copy(x)
        it += 1
    return x, it


def gauss_seidel_rel_method(A, v, w, eps):
    # eps es el error de convergencia
    N = len(v)
    # Creamos nuestra descompsición D,L,U
    DL = np.tril(A)  # tril crea un matriz triangular inferior
    U = A - DL  # creamos la matrix U
    x0 = np.zeros(N, float)  # creamos la primera estimación de x

    # inicializamos x

    err = 1e6
    it = 0
    while err > eps:
        x = (v - np.dot(U, x0))
        # Sustitución hacía adelante
        for m in range(N):
            for i in range(m):
                x[m] -= DL[m, i] * x[i]
            x[m] = (1 - w) * x0[m] + w / DL[m, m] * x[m]
        err = max(abs(x - x0))
        x0 = np.copy(x)
        it += 1
    return x, it
