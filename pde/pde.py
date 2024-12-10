import numpy as np

"""
• Elípticas (Laplace, Poisson), AC − B2 > 0.
• Hiperbólicas (ecuación de ondas), AC − B2 < 0.
• Parabólicas (ecuación del calor o difusión) AC − B2 = 0.
"""

#---------------------------------------------Equations with boundary conditions----------------------------------------

def jacobi_laplace(phi, eps):
    """
    :param phi: Initial estimation of the potential field.
    :param eps: Tolerance for convergence.
    :return:
        - phi: Final potential distribution.
        - it: Number of iterations performed.
    """
    if phi.ndim == 2:
        M = len(phi) - 1
        delta = 1.0
        phi_prime = np.copy(phi)
        it = 0
        while delta > eps:
            it += 1
            phi_prime[1:M, 1:M] = (phi[0:M - 1, 1:M] + phi[2:M + 1, 1:M] + phi[1:M, 0:M - 1] + phi[1:M, 2:M + 1]) / 4
            delta = np.max(abs(phi - phi_prime))
            phi = np.copy(phi_prime)
    elif phi.ndim == 3:
        print('Hola')
    return phi, it

def jacobi_poisson(phi,L,rho, eps):
    """
    :param phi: Initial estimation of the potential field.
    :param L: Length of the domain.
    :param rho: Density charge.
    :param eps: Tolerance for convergence.
    :return:
        - phi: Final potential distribution.
        - it: Number of iterations performed.
    """
    if phi.ndim == 2:
        M = len(phi) - 1
        a = L / M
        epsilon0 = 8.85e-12
        delta = 1.0
        phi_prime = np.copy(phi)
        it = 0
        while delta > eps:
            it += 1
            phi_prime[1:M, 1:M] = ((phi[2:M + 1, 1:M] + phi[0:M - 1, 1:M] + phi[1:M, 0:M - 1] + phi[1:M, 2:M + 1]) / 4
                                  + rho[1:M, 1:M] * a * a / (4 * epsilon0))
            delta = np.max(abs(phi - phi_prime))
            phi = np.copy(phi_prime)
    elif phi.ndim == 3:
        M = len(phi) - 1
        a = L / M
        epsilon0 = 8.85e-12
        delta = 1.0
        phi_prime = np.copy(phi)
        it = 0
        while delta > eps:
            it += 1
            phiprime[1:M, 1:M, 1:M] = ((phi[0:M - 1, 1:M, 1:M] + phi[2:M + 1, 1:M, 1:M] + phi[1:M, 0:M - 1, 1:M] +
                                        phi[1:M, 2:M + 1, 1:M] + phi[1:M, 1:M, 0:M - 1] + phi[1:M, 1:M, 2:M + 1]) / 6
                                       + rho[1:M, 1:M, 1:M] * a * a / (6 * epsilon0))
            delta = np.max(abs(phi - phi_prime))
            phi = np.copy(phi_prime)
    return phi, it


def gauss_seidel_overelaxed_laplace(phi, eps, w=0.9):
    """
    :param phi: Initial estimation of the potential field.
    :param w: Relaxation factor, which controls the convergence speed.
    :param eps: Tolerance for convergence.
    :return:
        - phi: Final potential distribution.
        - it: Number of iterations performed.

    """
    delta = 1.0
    M = len(phi) - 1
    it = 0
    while delta > eps:
        delta = 0
        it += 1
        # Calculamos los nuevos valores del potencial
        for i in range(1, M):
            for j in range(1, M):
                phip = phi[i, j]
                phi[i, j] = (1 + w) * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1]) / 4 - w * phi[
                    i, j]
                diff = abs(phi[i, j] - phip)
                delta = max([delta, diff])
    return phi, it

# -----------------------------------------Equations with initial conditions--------------------------------------------

