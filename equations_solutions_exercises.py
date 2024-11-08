import numpy as np
#------------------------------Linear equations solutions--------------------------------------------------------------
import linear_equations_solutions
from linear_equations_solutions import gaussian_elimination, gaussian_elimination_pivot, LU_factorization, LU_solution

""""
Ejemplo 5.1: Eliminación gaussiana
Usar el metodo de la eliminación de Gauss para transformar el sistema de ecuaciones:
    2w + x + 4 y + z = - 4,
    3w + 4 x - y - z =3,
    w - 4 x + y + 5 z =9,
    2w - 2 x + y + 3 z =7.
"""
# Definimos nuestra matriz ampliada
A=np.array([[2,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
B=np.array([-4,3,9,7],float)

x, AB2 = gaussian_elimination(A,B)
AB=np.column_stack((A,B))
print("Nuestra matriz extendida inicial")
print(AB)
print("")
print("Nuestra matriz extendida triangular")
print(AB2)
print("")
print("La solución del sistema de ecuaciones es:")
print(x)
print("")
print("Comprobamos que es solución")
print(np.dot(A,x)-B)

""""
Ejemplo 5.2: Eliminación gaussiana incluyendo el pivote
Resolver el sistema incluyendo el pivote

    x + 4 y + z = - 4,
    3w + 4 x - y - z =3,
    w - 4 x + y + 5 z =9,
    2w - 2 x + y + 3 z =7.

"""
# Definimos nuestra matriz ampliada
A=np.array([[0,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
B=np.array([-4,3,9,7],float)

x, AB2 = gaussian_elimination_pivot(A,B)

AB=np.column_stack((A,B))
print("Nuestra matriz extendida inicial")
print(AB)
print("")
print("Nuestra matriz extendida triangular")
print(AB2)
print("")
print("La solución del sistema de ecuaciones es:")
print(x)
print("")
print("Comprobamos que es solución")
print(np.dot(A,x)-B)

""""
Ejemplo 5.5: Aplicando la descomposición LU 
"""
A=np.array([[2,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
V=np.array([-4,3,9,7],float)
# Generamos las matrices L y U y comprobamos que son como deberían ser.
L,U=LU_factorization(A)
print(L)
print()
print(U)
print()
# Comprobamos que L U es efectivamente A.
print(np.dot(L,U))
print()
print(A)
# Solución del sistema usando LU
print("Solución del sistema usando LU",LU_solution(A,V))
#--------------------------------------Eigenvalues and eigenvectors-----------------------------------------------------







#-----------------------------------------------------Non linear equations----------------------------------------------










#----------------------------------------------------Optimization-------------------------------------------------------