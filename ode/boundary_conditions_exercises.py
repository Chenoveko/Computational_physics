import numpy as np
import matplotlib.pyplot as plt
from ode import *

"""
Ejemplo 7.12: Lanzamiento proyectil vertical
Sea una pelota lanzada en vertical desde el suelo. Resolver el problema de contorno sabiendo que
la pelota debe alcanzar el suelo al cabo de 10 s.

    d^2x/dt^2 = -g
"""
g = 9.81  # aceleración de la gravedad
a = 0.0  # tiempo inicial
b = 10.0  # tiempo final
N = 1000  # Número de pasos de Runge-Kutta
h = (b - a) / N  # tamaño de los pasos de Runge-Kutta
eps = 1e-10  # precisión requerida en la busqueda por bisección
v1 = 0.01
v2 = 1000.0


def proyectil(r):
    x = r[0]
    vx = r[1]
    fx = vx
    fvx = -g
    return np.array([fx, fvx], float)
tp, xp, vp,v = shoot(proyectil, a, b, v1, v2, N, eps)
print("La velocidad inicial buscada es %0.2f" %v,"m/s")

plt.plot(tp,xp)
plt.title('Proyectil vertical')
plt.xlabel('t')
plt.ylabel('x')
plt.axhline(0, color='k', linestyle=':')
plt.show()


"""
Ejemplo 7.13: El estado fundamental de un pozo cuadrado
"""
m = 9.1094e-31  # Masa del electrón
hbar = 1.0546e-34  # h barra
e = 1.6022e-19  # carga del electrón
L = 5.2918e-11  # radio de Bhor
N = 1000
h = L / N


def V1(x):
    return 0.0


def V2(x):
    V0 = 100 * e
    return V0 * x / L * (x / L - 1)


def schrodinguer(r, x, E):
    psi = r[0]
    phi = r[1]
    fpsi = phi
    fphi = (2 * m / hbar ** 2) * (V1(x) - E) * psi
    return np.array([fpsi, fphi], float)


# función que resuelve la ecuación de Schrödinger para una energía E.
def solve_schrodinguer_equation(E):
    psi = 0.0
    phi = 1.0
    r = np.array([psi, phi], float)
    for x in np.arange(0, L, h):
        k1 = h * schrodinguer(r, x, E)
        k2 = h * schrodinguer(r + 0.5 * k1, x + 0.5 * h, E)
        k3 = h * schrodinguer(r + 0.5 * k2, x + 0.5 * h, E)
        k4 = h * schrodinguer(r + k3, x + h, E)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return r[0]


def fundamental_energy():
    # Programa principal para encontrar la energía por el metodo de la secante
    E1 = 0
    E2 = 100 * e
    psi2 = solve_schrodinguer_equation(E1)
    eps = e / 1000
    while abs(E1 - E2) > eps:
        psi1, psi2 = psi2, solve_schrodinguer_equation(E2)
        E1, E2 = E2, E2 - psi2 * (E2 - E1) / (psi2 - psi1)
    print("El estado fundamental tiene una energía E = %0.2f" % (E2 / e), "eV.")
    return E2


fundamental_energy()

print(solve_schrodinguer_equation(fundamental_energy()))
