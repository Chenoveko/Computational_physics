import numpy as np


# ---------Gaussian elimination--------------#

def gaussian_elimination(A: 'Coeficients matrix', B: 'Column vector') -> 'Solution of the system using gaussian elimination':
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


# use case for the gaussian elimination
A = np.array([[2, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)
B = np.array([-4, 3, 9, 7], float)
AB = np.column_stack((A, B))
print("Nuestra matriz extendida inicial")
print(AB)
print("")
x, AB2 = gaussian_elimination(A, B)
print("Nuestra matriz extendida triangular")
print(AB2)
print("")
print("La solución del sistema de ecuaciones es:")
print(x)
print("")
print("Comprobamos que es solución")
print(np.dot(A, x) - B)


# -----------------Gaussian elimination with pivot--------------------

def gaussian_elimination_pivot(A: 'Coeficients matrix', B: 'Column vector') -> 'Solution of the system using gaussian elimination with pivot':
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


# use of case for the gaussian elimination with pivote
A = np.array([[0, 1, 4, 1], [3, 4, -1, -1], [1, -4, 1, 5], [2, -2, 1, 3]], float)
B = np.array([-4, 3, 9, 7], float)
AB = np.column_stack((A, B))
print("Nuestra matriz extendida inicial")
print(AB)
print("")
x, AB2 = gaussian_elimination_pivot(A, B)
print("Nuestra matriz extendida triangular")
print(AB2)
print("")
print("La solución del sistema de ecuaciones es:")
print(x)
print("")
print("Comprobamos que es solución")
print(np.dot(A, x) - B)


#---------------------------LU factorization------------------------------

def LU_factorization(A: 'Matrix to factorize') -> 'Lower Upper factorization of the matrix A':
    N = len(A)
    # Inicializamos nuestras dos matrices.
    L = np.zeros([N, N], float)  # L tiene que ser una matriz triangular inferior.
    U = np.copy(A)  # U será la matriz A convertida en triangular superior
    # Creamos la matrix L
    for m in range(N):
        L[m:N, m] = U[m:N, m]  # para cada iteracción (para cada fila), la columna m de L queda fijada por el valor que tiene U.
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

# use case for the LU factorization
A=np.array([[2,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
# Generamos las matrices L y U y comprobamos que son como deberían ser.
L,U=LU_factorization(A)
print('L = ',L)
print()
print('U = ',U)
print()
# Comprabamos que L U es efectivamente A.
print('Producto de L*U = ',np.dot(L,U))
print()
print('Es igual a A',A)
print()

#---------------------------LU method------------------------------

def LU_solution(A: 'Coeficients matrix', V: 'Column vector') -> 'Solution of the system using LU method':
    N = len(A)
    L,U = LU_factorization(A)
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

# use case for the LU solving method
A=np.array([[2,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
V=np.array([-4,3,9,7],float)
print(LU_solution(A,V))
print()
# -------------------------- Solve a linear matrix equation, or system of linear scalar equations using numpy---------

def numpy_solver(A: 'Coeficients matrix', B: 'Column vector') -> 'Solution of the system using numpy method':
     return np.linalg.solve(A,B)

# use case for the numpy solving method
A=np.array([[0,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
V=np.array([-4,3,9,7],float)
print(numpy_solver(A,V))
print()