import numpy as np
from eigenvalues_eigenvectors import *

"""
Ejemplo 5.12: estudiando la descomposición QR
"""
A = np.array([[1, 4, 8, 4], [4, 2, 3, 7], [8, 3, 6, 9], [4, 7, 9, 2]], float)

Q, R = QR_factorization(A)
print("Q = ", Q)
print()
print("R = ", R)
print()
print(np.dot(Q, R) - A)

"""
Ejemplo 5.13: calculando autovalores y autovectores de la matriz A
"""
D, V = QR_diagonalization(A, 1e-12)

print("D = ", D)
print()
print("V = ", V)
print()
print("Nuestros autovalores son: ", np.diag(D))

# Comprobemos que son autovalores y autovectores:
for i in range(len(A)):
    print(np.dot(A, V[:, i]) - np.diag(D)[i] * V[:, i])
