import numpy as np
import matplotlib.pyplot as plt
from ode import *

"""
Ejemplo 7.1: aplicando el metodo de Euler en el intervalo [0, 10] usando 1000 puntos equiespaciados 
con la condición inicial x(t = 0) = 0

    dx/dt =-x^3 + sin(t)
"""


def f(x, t):
    return -x ** 3 + np.sin(t)


a = 0  # comienzo del intervalo
b = 10  # final del intervalo
N = 1000  # número de pasos
x0 = 0  # condición inicial

tp, xp = euler(f, x0, a, b, N)
plt.plot(tp, xp)
plt.title("Euler method")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

"""
Ejemplo 7.2: aplicando el metodo de Runge-Kutta de orden dos en el intervalo [0, 10] usando 100 puntos equiespaciados 
con la condición inicial x(t = 0) = 0
Compararlo con el resultado obtenido con el método de Euler para 100 pasos.

    dx/dt =-x^3 + sin(t)
"""

tpp, xpp = runge_kutta_2order(f, x0, a, b, N)
plt.plot(tp, xp, label="Euler")
plt.plot(tpp, xpp, label="Runge-Kutta second order")
plt.legend()
plt.title("Euler Method and Runge-Kutta 2 method")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

"""
Ejemplo 7.3: aplicando el metodo de Runge-Kutta de cuarto orden en el intervalo [0, 10] usando 50 puntos equiespaciados 
con la condición inicial x(t = 0) = 0
Compararlo con el método de Euler y el método de Runge-Kutta de orden dos para N = 100.

    dx/dt =-x^3 + sin(t)
"""

tppp, xppp = runge_kutta_4order(f, x0, a, b, N)
plt.plot(tp, xp, label="Euler")
plt.plot(tpp, xpp, label="Runge-Kutta second order")
plt.plot(tppp, xppp, 'o', label="Runge-Kutta forth order")
plt.legend()
plt.title("Euler Method and Runge-Kutta 2,4 method")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

"""
Ejemplo 7.4 Ecuaciones diferenciales en rangos infinitos t e[o, infinito]: usar el método de Runge-Kutta de cuarto orden para obtener
 su solución en el intervalo t 2 [0,inf) con la condición inicial x(0)=1. Representarla en el intervalo [0,100].

    dx/dt = 1/(x^2+t^2)

Haciendo el cambio de variable t = u/(1-u) tenemos:

    dx/dt = 1/[(1-u)^2*x^2+u^2] = g(x,u), con u perteneciente a [0, 1].
"""


def g(x, u):
    return 1 / (x ** 2 * (1 - u) ** 2 + u ** 2)


N = 100  # número de puntos
x0 = 1  # condición inicial

tp, xp = runge_kutta_4order_infinite_range(g, x0, N)
plt.plot(tp, xp)
plt.xlim(1, 100)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("EDO rangos infinitos")
plt.show()
