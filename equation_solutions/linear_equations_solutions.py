import numpy as np


# -----------------------------------------Gauss Method------------------------------------------------------------------
def gaussian_elimination(A, B):
    """
    :param A: Coefficients matrix of the system of equations.
    :param B: Column vector of the system of equations.
    :return: A tuple consisting of two elements:
        - x: Solution of the system using Gaussian elimination.
        - AB: Augmented triangular matrix.
    """
    N = len(B)
    AB = np.column_stack((A, B))
    for m in range(N):
        div = AB[m, m]
        AB[m, :] /= div
        for i in range(m + 1, N):
            mult = AB[i, m]
            AB[i, :] -= mult * AB[m, :]
        x = np.zeros(N, float)
        for m in range(N - 1, -1, -1):
            x[m] = AB[m, N]
            for i in range(m + 1, N):
                x[m] -= AB[m, i] * x[i]
    return x, AB


def gaussian_elimination_pivot(A, B):
    """
    :param A: Coefficients matrix of the system of equations.
    :param B: Column vector of the system of equations.
    :return: A tuple consisting of two elements:
        - x: Solution of the system using Gaussian elimination with pivot.
        - AB: Augmented triangular matrix.
    """
    N = len(B)
    AB = np.column_stack((A, B))
    for m in range(N):
        pivot = m
        largest = abs(AB[m, m])
        for i in range(m + 1, N):
            if abs(AB[i, m] > largest):
                largest = AB[i, m]
                pivot = i
        if pivot != m:
            for i in range(N + 1):
                AB[m, i], AB[pivot, i] = AB[pivot, i], AB[m, i]
        div = AB[m, m]
        AB[m, :] /= div
        for i in range(m + 1, N):
            mult = AB[i, m]
            AB[i, :] -= mult * AB[m, :]
    x = np.zeros(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = AB[m, N]
        for i in range(m + 1, N):
            x[m] -= AB[m, i] * x[i]
    return x, AB


# ----------------------------------------------LU method----------------------------------------------------------------

def LU_factorization(A):
    """
    :param A: Matrix to factorize.
    :return: A tuple consisting of two elements:
        - L: Lower factorization of the matrix A.
        - U: Upper factorization of the matrix A.
    """
    N = len(A)
    L = np.zeros([N, N], float)
    U = np.copy(A)
    for m in range(N):
        L[m:N, m] = U[m:N, m]
        div = U[m, m]
        U[m, :] /= div
        for i in range(m + 1, N):
            mult = U[i, m]
            U[i, :] -= mult * U[m, :]
    return L, U


def LU_solve(A, B):
    """
    :param A: Coefficients matrix of the system of equations.
    :param B: Column vector of the system of equations.
    :return x: Solution of the system using LU method
    """
    N = len(A)
    L, U = LU_factorization(A)
    y = np.empty(N, float)
    for m in range(N):
        y[m] = B[m]
        for i in range(m):
            y[m] -= L[m, i] * y[i]
        y[m] /= L[m, m]
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = y[m]
        for i in range(m + 1, N):
            x[m] -= U[m, i] * x[i]
    return x


def LU_pivot_factorization(A):
    """
    :param A: Matrix to factorize.
    :return: A tuple consisting of three elements:
        - L: Lower triangular matrix
        - U: Upper triangular matrix
        - P: Permutation vector indicating the pivoting changes applied during the decomposition process.
    """
    N = len(A)
    L = np.zeros([N, N], float)
    U = np.copy(A)
    P = np.empty(N, int)
    for m in range(N):
        L[m:N, m] = U[m:N, m]
        pivot = m
        largest = abs(U[m, m])
        for i in range(m + 1, N):
            if abs(U[i, m]) > largest:
                largest = abs(U[i, m])
                pivot = i
        for i in range(N):
            U[m, i], U[pivot, i] = U[pivot, i], U[m, i]
            L[m, i], L[pivot, i] = L[pivot, i], L[m, i]
        P[m] = pivot
        div = U[m, m]
        U[m, :] /= div
        for i in range(m + 1, N):
            mult = U[i, m]
            U[i, :] -= mult * U[m, :]
    return L, U, np.array(P)


def LU_pivot_solve(A, B):
    """
    :param A: Coefficients matrix of the system of equations.
    :param B: Column vector of the system of equations.
    :return x: Solution of the system using LU method with pivot
    """
    N = len(B)
    L, U, P = LU_pivot_factorization(A)
    V = np.copy(B)
    for m in range(N):
        pivot = P[m]
        V[m], V[pivot] = V[pivot], V[m]
    y = np.empty(N, float)
    for m in range(N):
        y[m] = V[m]
        for i in range(m):
            y[m] -= L[m, i] * y[i]
        y[m] /= L[m, m]
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = y[m]
        for i in range(m + 1, N):
            x[m] -= U[m, i] * x[i]
    return x


def LU_inverse(A):
    """
    :param A: Matrix to invert.
    :return A_inv: The inverse of the matrix A using LU decomposition.
    """
    N = len(A)
    A_inv = np.zeros([N, N], float)
    for i in range(N):
        I = np.zeros(N, float)
        I[i] = 1
        A_inv[:, i] = LU_solve(A, I)
    return A_inv


# ---------------------------------------------Iterative methods for linear equations------------------------------------

def jacobi(A, B, eps):
    """
    :param A: Coefficients matrix of the system of equations.
    :param B: Column vector of the system of equations.
    :param eps: Convergence tolerance, specifies the precision of the solution.
    :return: A tuple consisting of two elements:
        - x: Solution of the system using Jacobi method.
        - it: Number of iterations taken to reach the specified precision.
    """
    N = len(B)
    D = np.diag(A)
    LU = A - np.diagflat(D)
    x0 = np.zeros(N, float)
    err = 1e6
    it = 0
    while err > eps:
        x = (B - np.dot(LU, x0)) / D
        err = max(abs(x - x0))
        x0 = np.copy(x)
        it += 1
    return x, it


def gauss_seidel(A, B, eps):
    """
    :param A: Coefficients matrix of the system of equations.
    :param B: Column vector of the system of equations.
    :param eps: Convergence tolerance, specifies the precision of the solution.
    :return: A tuple consisting of two elements:
        - x: Solution of the system using Gauss-Seidel method.
        - it: Number of iterations taken to reach the specified precision
    """
    N = len(B)
    DL = np.tril(A)
    U = A - DL
    x0 = np.zeros(N, float)
    err = 1e6
    it = 0
    while err > eps:
        x = (B - np.dot(U, x0))
        for m in range(N):
            for i in range(m):
                x[m] -= DL[m, i] * x[i]
            x[m] /= DL[m, m]
        err = max(abs(x - x0))
        x0 = np.copy(x)
        it += 1
    return x, it


def gauss_seidel_rel(A, B, w, eps):
    """
    :param A: Coefficients matrix of the system of equations.
    :param B: Column vector of the system of equations.
    :param w: Relaxation factor for the iterative method.
    :param eps: Convergence tolerance, specifies the precision of the solution.
    :return: A tuple consisting of three elements:
        - x: Solution of the system using Gauss-Seidel over-relaxation method.
        - it: Number of iterations taken to reach the specified precision.
    """
    N = len(B)
    DL = np.tril(A)
    U = A - DL
    x0 = np.zeros(N, float)
    err = 1e6
    it = 0
    while err > eps:
        x = (B - np.dot(U, x0))
        for m in range(N):
            for i in range(m):
                x[m] -= DL[m, i] * x[i]
            x[m] = (1 - w) * x0[m] + w / DL[m, m] * x[m]
        err = max(abs(x - x0))
        x0 = np.copy(x)
        it += 1
    return x, it
