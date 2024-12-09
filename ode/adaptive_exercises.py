import numpy as np
from ode import *
import matplotlib.pyplot as plt

"""
Ejemplo 7.8: la órbita de un cometa
Resolver el problema con paso fijo y paso adaptado

    d^2x/dt^2 = -GMx/r^3
    d^2y/dt^2 = -GMy/r^3
"""
# constantes del problema
G = 6.674e-11
M = 1.989e30


# definimos la función de nuestro sistema de ecuaciones diferenciales
def orbita_cometa(r):
    x = r[0]
    y = r[1]
    vx = r[2]
    vy = r[3]
    r3 = (x ** 2 + y ** 2) ** 1.5
    fx = vx
    fy = vy
    fvx = -G * M * x / r3
    fvy = -G * M * y / r3
    return np.array([fx, fy, fvx, fvy], float)


# Metodo de Runge-Kutta de paso fijo
x0 = 4e12
y0 = 0
vx0 = 0
vy0 = 500
r0 = np.array([x0, y0, vx0, vy0], float)  # condiciones iniciales
tmax = 3.2e9  # tiempo total simulacion
N = 100000

tp, xp, yp = runge_kutta_4order_system(orbita_cometa, r0, 0, tmax, N)

plt.plot(xp, yp)
plt.title("Orbita de cometa paso fijo")
plt.show()

# Metodo de Runge-Kutta de paso adaptado
delta = 1e3 / 365.25 / 24 / 3600
h0 = 1e5
tp, xp, yp = runge_kutta_4order_system_adaptive(orbita_cometa, r0, tmax, delta, h0)

plt.plot(xp, yp)
plt.title("Orbita de cometa paso adaptado")
plt.show()