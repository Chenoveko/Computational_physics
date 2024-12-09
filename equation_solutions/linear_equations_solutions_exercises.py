import numpy as np
from linear_equations_solutions import *

""""
Ejemplo 5.1: Eliminación gaussiana
Usar el metodo de la eliminación de Gauss para transformar el sistema de ecuaciones:
    2w + x + 4 y + z = - 4,
    3w + 4 x - y - z =3,
    w - 4 x + y + 5 z =9,
    2w - 2 x + y + 3 z =7.
"""
# Definimos nuestra matriz ampliada
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)
B = np.array([-4, 3, 9, 7], float)

x, AB2 = gaussian_elimination(A, B)

print('-----------------------------------Ejemplo 5.1: Eliminación gaussiana------------------------------------------')
print("Nuestra matriz extendida inicial")
print(np.column_stack((A, B)))
print("---")
print("Nuestra matriz extendida triangular")
print(AB2)
print("---")
print("La solución del sistema de ecuaciones es:")
print(x)
print("---")
print("Comprobamos que es solución")
print(np.dot(A, x) - B)

""""
Ejemplo 5.2: Eliminación gaussiana incluyendo el pivote
Resolver el sistema incluyendo el pivote

    x + 4 y + z = - 4,
    3w + 4 x - y - z =3,
    w - 4 x + y + 5 z =9,
    2w - 2 x + y + 3 z =7.
"""
# Definimos nuestra matriz ampliada
A = np.array([[0, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)
B = np.array([-4, 3, 9, 7], float)

x, AB2 = gaussian_elimination_pivot(A, B)

print('-------------------------------Ejemplo 5.2: Eliminación gaussiana con pivote-----------------------------------')
print("Nuestra matriz extendida inicial")
print(np.column_stack((A, B)))
print("---")
print("Nuestra matriz extendida triangular")
print(AB2)
print("---")
print("La solución del sistema de ecuaciones es:")
print(x)
print("---")
print("Comprobamos que es solución")
print(np.dot(A, x) - B)

""""
Ejemplo 5.5: Aplicando la descomposición LU al sistema del ejemplo 5.1
"""
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)
V = np.array([-4, 3, 9, 7], float)
# Generamos las matrices L y U y comprobamos que son como deberían ser.
L, U = LU_factorization(A)

print('------------------------------Ejemplo 5.5: Descomposición LU------------------------------')
print(L)
print()
print(U)
print()
# Comprobamos que L U es efectivamente A.
print(np.dot(L, U))
print()
print(A)
# Solución del sistema usando LU
print("Solución del sistema usando LU", LU_solve(A, V))

""""
Ejercicio 5.1: Aplicando la descomposición LU con pivote al sistema del ejemplo 5.1

La descomposición LU se usa en la función solve del módulo linalg del paquete numpy.
"""
A = np.array([[0, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)
B = np.array([-4, 3, 9, 7], float)

print('------------------------------Ejercicio 5.1: Descomposición LU con pivote------------------------------')
# Solución del sistema usando LU  con pivote
print("Solución del sistema usando LU con pivote", LU_pivot_solve(A, B))
print("Solución del sistema usando numpy.linalg.solve ", np.linalg.solve(A, B))

""""
Ejemplo 5.7: Encontrar la inversa de la matriz del ejercicio 5.1 usando LU
"""
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)

print('------------------------------Ejemplo 5.7: Inversa de una matriz usando LU------------------------------')
print("Matriz A ", A)

print("Inversa de A usando LU ", LU_inverse(A))
print("Inversa de A usando numpy.linalg.inv ", np.linalg.inv(A))
print("A * A ^-1 = ", np.dot(A, LU_inverse(A)))
print("A * A ^-1 = ", np.dot(A, np.linalg.inv(A)))

""""
Ejemplo 5.9: aplicando el método de Jacobi
"""
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)

print('------------------------Ejemplo 5.9: Solución de un sistema por el método de Jacobi-------------------------')

A = np.array([[8, 1, 2, 1], [1, 5, -1, -1], [1, -4, 6, 1], [1, -2, 1, 5]], float)
V = np.array([-1, 3, 2, 1], float)
x1, it = jacobi(A, V, 1e-6)
x2, AB = gaussian_elimination(A, V)
print(x1, x2, sep=", ")
print("El número de iteraciones del método de jacobi es: ", it)

""""
Ejemplo 5.10: aplicando el método de Gauss-Seidel
"""

print('----------------------Ejemplo 5.10: Solución de un sistema por el método de Gauss-Seidel-----------------------')

A = np.array([[8, 1, 2, 1], [1, 5, -1, -1], [1, -4, 6, 1], [1, -2, 1, 5]], float)
V = np.array([-1, 3, 2, 1], float)
x1, it = gauss_seidel(A, V, 1e-6)
x2, AB = gaussian_elimination(A, V)
print(x1, x2, sep=", ")
print("El número de iteraciones del método de Gauss-Seidel es: ", it)

""""
Ejemplo 5.11: aplicando el método de Gauss-Seidel sobrerrelajado
"""

print('-------------Ejemplo 5.11: Solución de un sistema por el método de Gauss-Seidel sobrerrelajadado---------------')

A = np.array([[8, 1, 2, 1], [1, 5, -1, -1], [1, -4, 6, 1], [1, -2, 1, 5]], float)
V = np.array([-1, 3, 2, 1], float)
w = 1.1
x1, it = gauss_seidel_rel(A, V, w, 1e-6)
x2, AB = gaussian_elimination(A, V)
print(x1, x2, sep=", ")
print("El número de itracciones del método de la sobrerelajación sucesiva es: ", it)
