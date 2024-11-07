import numpy as np
import matplotlib.pyplot as plt
from differential_equations import *

# -----------------------------------------First order ODE--------------------------------------------------------------
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

tp, xp = euler_method(f, x0, a, b, N)
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

tpp, xpp = runge_kutta_2_method(f, x0, a, b, N)
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

tppp, xppp = runge_kutta_4_method(f, x0, a, b, 50)
plt.plot(tp, xp, label="Euler")
plt.plot(tpp, xpp, label="Runge-Kutta second order")
plt.plot(tppp, xppp, 'o', label="Runge-Kutta forth order")
plt.legend()
plt.title("Euler Method and Runge-Kutta 2,4 method")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

"""
Ejemplo 7.4 Ecuaciones diferenciales en rangos infinitos: usar el método de Runge-Kutta de cuarto orden para obtener
 su solución en el intervalo t 2 [0,inf) con la condición inicial x(0)=1. Representarla en el intervalo [0,100].

    dx/dt = 1/(x^2+t^2)
    
Haciendo el cambio de variable t = u/(1-u) tenemos:

    dx/dt = 1/[(1-u)^2*x^2+u^2] = g(x,u), con u perteneciente a [0, 1].
"""


def g(x, u):
    return 1 / (x ** 2 * (1 - u) ** 2 + u ** 2)


a = 0.0  # límite inferior en u
b = 0.99999  # límite superior en u
N = 100  # número de puntos
x0 = 1  # condición inicial

tp, xp = runge_kutta_4_infinite_range_method(g, x0, a, b, N)
plt.plot(tp, xp)
plt.xlim(1, 100)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
# -----------------------------------------ODE systems-------------------------------------------------------------------
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
r0 = np.array([1, 1], float)  # condiciones iniciales
N = 1000  # número de puntos

tp, xp, yp = runge_kutta_4_system_method(ode_system, r0, a, b, N)
plt.plot(tp, xp)
plt.plot(tp, yp)
plt.title("Ode system")
plt.xlabel("t")
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


def lokta_volterra(r):
    alfa, beta, gamma, delta = 1, 0.5, 0.5, 2
    x = r[0]
    y = r[1]
    fx = alfa * x - beta * x * y
    fy = gamma * x * y - delta * y
    return np.array([fx, fy], float)


a = 0  # punto inicial del intervalo
b = 30  # punto final del intervalo
r0 = np.array([2, 2], float)  # condiciones iniciales
N = 10000  # número de puntos

tp, xp, yp = runge_kutta_4_system_method(lokta_volterra, r0, a, b, N)
plt.plot(tp, xp, label="Población de conejos")
plt.plot(tp, yp, label='Población de zorros')
plt.title("Lokta-Volterra system")
plt.xlabel("t")
plt.ylabel("Miles de individuos")
plt.legend()
plt.show()
# -----------------------------------------Second order ODE-------------------------------------------------------------
