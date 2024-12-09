import numpy as np
from ode import *
import matplotlib.pyplot as plt

"""
Ejemplo 7.19: Método del salto de rana

    d^2x/dt^2 = (dx/dt)^2 - x - 5

usando el método del salto de la rana en el intervalo t e [0, 50], con un paso h=0.001 y usando las
condiciones iniciales x(0) = 1, v(0) = 0.
"""


def f_rana(r):
    x = r[0]
    v = r[1]
    fr = v
    fv = v ** 2 - x - 5
    return np.array([fr, fv], float)


# intervalo y paso
a = 0
b = 50
h = 0.001
r1 = np.array([1, 0], float)

tp, xp = leapfrog(f_rana, r1, a, b, h)

plt.plot(tp, xp)
plt.title("Leapfrog method")
plt.xlabel("t")
plt.ylabel("x")
plt.show()

"""
Ejemplo 7.10: el método de Verlet y la órbita de la tierra
    
        d^2r/dt^2 = -G*M*r/|r|^3
"""

# Constantes
G = 6.6738e-11
M = 1.9891e30
m = 5.9722e24


def orbita_tierra(r):
    return -G * M * r / np.linalg.norm(r) ** 3


def energia_potencial(r):
    return -G * M * m / np.linalg.norm(r)


def energia_cinetica(v):
    return 0.5 * m * sum(v * v)


# Intervalo temporal
a = 0.0
b = 100e6
h = 3600.0

# Condiciones iniciales
x0 = 1.4710e11
y0 = 0.0
vx0 = 0.0
vy0 = 3.0278e4
r = np.array([x0, y0], float)
v = np.array([vx0, vy0], float)

tp, xp, yp, Vp, Tp, Ep = verlet(orbita_tierra, energia_cinetica, energia_potencial, r, v, a, b, h)

plt.plot(xp, yp)
plt.title("órbita de tierra con Verlet")
plt.show()

plt.plot(tp, Vp, label='Energía potencial')
plt.plot(tp, Ep, label='Energía total')
plt.plot(tp, Tp, label='Energía cinética')
plt.title('Orbita tierra')
plt.legend()
plt.show()

plt.plot(tp, Ep, 'g')
plt.title('Energía total')
plt.show()

"""
Ejemplo 7.11: el método de Bulirsch-Stoer y el péndulo no-lineal
"""
g = 9.81
l = 0.1

# definimos la precisión objetivo por unidad de tiempo
delta = 1e-8


def pendulo(r):
    theta = r[0]
    omega = r[1]
    ftheta = omega
    fomega = -(g / l) * np.sin(theta)
    return np.array([ftheta, fomega], float)


# intervalo para la solución
a = 0
b = 10
N = 100

# condiciones iniciales
r0 = np.array([179 * np.pi / 180, 0], float)

tp, xp = bulirsch_stoer(pendulo, r0, a, b, N, delta)

plt.plot(tp, xp)
plt.title("Bulirsch-Stoer method")
plt.show()
