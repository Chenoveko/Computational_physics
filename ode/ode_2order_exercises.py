import numpy as np
from fontTools.merge import computeMegaGlyphOrder

from ode import *
import matplotlib.pyplot as plt

"""
Ejercicio 7.6: el péndulo no lineal
Escribir un programa que permita describir el movimiento de un péndulo no lineal durante
4 periodos, con l = 10 cm que parte del reposo con un ángulo inicial de q = 179º usando el
metodo de Runge-Kutta de cuarto orden.

    d^2theta/dt^2 = -g/l * sin(theta) -> lo convertimos en un sistema

    dtheta/dt = omega
    domega/dt = -g/l * sin(theta)
"""


def pendulo(r):
    theta = r[0]
    omega = r[1]
    f_theta = omega
    f_omega = -g / l * np.sin(theta)
    return np.array([f_theta, f_omega], float)


a = 0
b = 10
N = 1000
g, l = 9.81, 0.1
r0 = np.array([np.deg2rad(179), 0], float)  # condiciones iniciales
tp, thetap, omegap = runge_kutta_4order_system(pendulo, r0, a, b, N)
xp = l * np.sin(thetap)
yp = -l * np.cos(thetap)

plt.plot(tp, thetap, "red")
plt.plot(tp, thetap, "b.")
plt.title("Pendulo no lineal")
plt.xlabel("t")
plt.ylabel("$\\theta$")
plt.show()

plt.plot(tp, omegap, "red")
plt.plot(tp, omegap, "b.")
plt.title("Pendulo no lineal")
plt.xlabel("t")
plt.ylabel("$\\omega$")
plt.show()

plt.plot(xp, yp)
plt.title("Pendulo no lineal")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

"""
Ejercicio 7.3: Péndulo forzado

    d^2theta/dt^2 = -g/l * sin(theta) + C * cos(theta) * sin(Omega*t) -> lo convertimos en un sistema

    dtheta/dt = omega
    domega/dt = -g/l * sin(theta)
"""

"""
1) Escribir un programa que resuelva la ecuación del movimiento de un péndulo forzado con
l = 10 cm, C = 2 s−2 y Ω = 5 s−1 y que represente su posición en el intervalo t ∈ [0, 100] s
para las condiciones iniciales θ = dθ/dt = 0.
"""

C, Omega = 2, 5


def pendulo_forzado(r, t):
    theta = r[0]
    omega = r[1]
    f_theta = omega
    f_omega = -g / l * np.sin(theta) + C * np.cos(theta) * np.sin(Omega * t)
    return np.array([f_theta, f_omega], float)


a = 0
b = 100
N = 1000
g, l = 9.81, 0.1
r0 = np.array([0, 0], float)  # condiciones iniciales
tp, thetap, omegap = runge_kutta_4order_system(pendulo_forzado, r0, a, b, N)
xp = l * np.sin(thetap)
yp = -l * np.cos(thetap)

plt.plot(tp, thetap, "red")
plt.plot(tp, thetap, "b.")
plt.title("Pendulo forzado")
plt.xlabel("t")
plt.ylabel("$\\theta$")
plt.show()

plt.plot(tp, omegap, "red")
plt.plot(tp, omegap, "b.")
plt.title("Pendulo forzado")
plt.xlabel("t")
plt.ylabel("$\\omega$")
plt.show()

plt.plot(xp, yp)
plt.title("Pendulo forzado")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

"""
Ejemplo 7.7: orbitando en el espacio
Ecuaciones de movimiento:

    d^2x/dt^2 = -G*M * x/(r^2sqrt(r^2+L^2/4))
    d^2y/dt^2 = -G*M * y/(r^2sqrt(r^2+L^2/4))
    
Resolver las ecuaciones en unidades en las que G = 1, M = 10, L = 2, usando las condiciones iniciales (x, y) = (1, 0) y
velocidad 1 en la dirección y. Calcular la órbita en el intervalo t ∈ [0, 10] y representar la trayectoria
en el plano xy. ¿Por qué el movimiento es diferente al de un planeta orbitando alrededor del Sol?
"""

# constantes
M = 10
L = 2
G = 1


def orbita(r):
    x = r[0]
    y = r[1]
    vx = r[2]
    vy = r[3]
    fx = vx
    fy = vy
    r2 = x ** 2 + y ** 2
    denom = r2 * np.sqrt(r2 + L ** 2 / 4)
    fvx = -M * G * x / denom
    fvy = -M * G * y / denom
    return np.array([fx, fy, fvx, fvy], float)


a = 0
b = 10
r0 = np.array([1, 0, 0, 1], float)
N = 1000

tp, xp, yp = runge_kutta_4order_system(orbita, r0, a, b, N)

plt.plot(xp, yp)
plt.title("Orbita")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
