import numpy as np

# ----------------------------------Descomposición QR--------------------------------------------------------------------
"""
El método más común usado para calcular los autovalores de una matriz es la descomposición QR.
"""


def QR_factorization(A):
    N = len(A)
    # Inicializamos los tres vectores que necesitamos para Gram-Schmidt
    U = np.zeros([N, N], float)
    Q = np.zeros([N, N], float)
    R = np.zeros([N, N], float)
    for m in range(N):
        U[:, m] = A[:, m]  # Creamos los vectores de U
        for i in range(m):
            R[i, m] = np.dot(Q[:, i], A[:, m])  # los vamos calculando en cada iteracción
            U[:, m] -= R[i, m] * Q[:, i]
        R[m, m] = np.linalg.norm(U[:, m])
        Q[:, m] = U[:, m] / R[m, m]
    return Q, R


def QR_diagonalization(A, eps):
    N = len(A)
    V = np.identity(N, float)  # Empezamos con la identidad
    delta = 1.0
    while delta > eps:
        # Hacemos un paso del algoritmo QR
        Q, R = QR_factorization(A)
        A = np.dot(R, Q)
        V = np.dot(V, Q)
        # Encontramos el mayor valor absoluto de los elementos fuera de la diagonal
        Ac = np.copy(A)
        for i in range(N):
            Ac[i, i] = 0.0
        delta = np.max(np.absolute(Ac))
    return A, V


# ----------------------------------------Autovalores y autovectores utilizando numpy------------------------------------
"""
En Python en el módulo numpy.linalg hay funciones para calcular las autofunciones y auto-
valores de una matriz A:

    • eigh calcula los autovalores y autovectores de un matriz simétrica o hermítica optimizando
      el algoritmo QR para este tipo de matrices.

    • eigvalsh calcula solo los autovalores. Devuelve un array con los autovalores.
    
    • Para matrices no simétricas o no hermíticas uno puede usar las funciones: eig y eigvals.
"""
