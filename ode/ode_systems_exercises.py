import numpy as np
from ode import *
import matplotlib.pyplot as plt

"""
Ejemplo 7.5: Sistemas de ecuaciones diferenciales ordinarias: encontrar su solución en el intervalo t e [0, 10],
con x(0)=y(0)=1.

    dx/dt = xy - x
    dy/dt = y -xy + sin(t)^2
"""


def ode_system(r, t):
    x = r[0]
    y = r[1]
    fx = x * y - x
    fy = y - x * y + np.sin(t) ** 2
    return np.array([fx, fy], float)


a = 0  # punto inicial del intervalo
b = 10  # punto final del intervalo
r0 = np.array([1, 1], float)  # condiciones iniciales  x(0)=y(0)=1
N = 1000  # número de puntos

tp, xp, yp = runge_kutta_4order_system(ode_system, r0, a, b, N)
plt.plot(tp, xp, label='x')
plt.plot(tp, yp, label='y')
plt.title("Ode system")
plt.xlabel("t")
plt.legend()
plt.show()

"""
Ejercicio 7.2: las ecuaciones de Lotka-Volterra
Escribir un programa que resuelva las ecuaciones usando el método de Runge-Kutta de
cuarto orden con alfa = 1, beta = gamma = 0,5 y delta = 2, usando las condiciones iniciales x(0)=y(0)=2 en
unidades de miles de individuos y representar sus poblaciones en el intervalo t e [0, 30].

    Población de conejos -> x , Población de zorros -> y
    dx/dt = alfa*x - beta*x*y
    dy/dt = gamma*x*y - delta*y
    todas las constantes son positivas
"""


def lotka_volterra(r):
    alfa, beta, gamma, delta = 1, 0.5, 0.5, 2
    x = r[0]
    y = r[1]
    fx = alfa * x - beta * x * y
    fy = gamma * x * y - delta * y
    return np.array([fx, fy], float)


a = 0  # punto inicial del intervalo
b = 30  # punto final del intervalo
r0 = np.array([2, 2], float)  # condiciones iniciales x(0)=y(0)=2
N = 10000  # número de puntos

tp, xp, yp = runge_kutta_4order_system(lotka_volterra, r0, a, b, N)
plt.plot(tp, xp, label="Población de conejos")
plt.plot(tp, yp, label='Población de zorros')
plt.title("Lokta-Volterra system")
plt.xlabel("t")
plt.ylabel("Miles de individuos")
plt.legend()
plt.show()
