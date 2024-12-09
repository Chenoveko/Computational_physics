import numpy as np
from non_linear_equations import *
from matplotlib import pyplot as plt

"""
Ejemplo 5.14: aplicando el metodo del punto fijo
Aplicar el método del punto fijo para encontrar la solución a la ecuación
    x = 2 - e^-x
que no tiene solución analítica
"""


def f(x):
    return 2 - np.exp(-x)


x0 = 1.0
eps = 1e-6
print('Ejemplo 5.14')
x1, it = fixed_point(f, x0, eps)
print('pto fijo en x = ', x1, ' se han necesitado ', it, ' iteraciones')

"""
Ejemplo 5.15: aplicando el método del punto fijo por segunda vez aplicar el método del punto
fijo para encontrar la solución a la ecuación
    x = e^(1-x^2)
"""


def g(x):
    return np.exp(1 - x ** 2)


x0 = 0
print('Ejemplo 5.15')
x1, it = fixed_point(g, x0, eps)

print('pto fijo en x = ', x1, ' se han necesitado ', it, ' max iteraciones')

"""
Ejemplo 5.16: aplicando el método del punto fijo por tercera vez
    x = x^2 + sin(2x) -> x = arcsin(x-x^2) * 1/2
"""

print('Ejemplo 5.16')


def h(x):
    return np.arcsin(x - x * x) / 2


x0 = 1e-4
x1, it = fixed_point(h, x0, eps)

print('pto fijo en x = ', x1, ' se han necesitado ', it, ' iteraciones')

"""
Ejercicio 5.5: Proceso de glucólisis
Los puntos estacionarios de nuestro modelo para glucólisis vienen dados por:
    0 = -x + ay + x^2y, 0 = b - ay - x2y.
donde a = 1 y b = 2 -> No converge
"""


def glucolisis(r):
    a, b = 1, 2
    x = r[1] * (a + r[0] ** 2)
    y = b / (a + r[0] ** 2)
    return [x, y]


x1 = fixed_point_multiple_variables_no_convergente(glucolisis, [[0, 0], [0, 0]])
print('Pto estacionario de glucólisis = ', x1)

""""
Ejemplo 5.17: visualizando el metodo de la biseccion. Encontrar la raíz de la función:
    f(x) = e^x - 2
"""


def j(x):
    return np.exp(x) - 2


xp = np.linspace(-2, 2, 100)
plt.plot(xp, j(xp))
plt.title('Visualizando el método de la bisección')
plt.show()

m, it = bisection(j, -2, 2, 1e-6)
print('Raíz en x = ', m)
print('Iteraciones del método bisección = ', it)

""""
Ejemplo 5.18: visualizando el metodo de la Newton-Raphson. Encontrar la raíz de la función:
    f(x) = e^x - 2
"""


def jp(x):
    return np.exp(x)


x1 = 2
raiz, it = newton_raphson(j, jp, x1, 1e-6)
print('Raíz en x = ', raiz)
print('Iteraciones del método Newton-Raphson = ', it)

"""
Ejemplo mio: Probando el método de la secante
"""

raiz, it = secant(j, -2, 2, 1e-6)
print('Raíz en x = ', raiz)
print('Iteraciones del método secante = ', it)

"""
Ejemplo 5.19: representación gráfica del método de Newton-Raphson en varias variables. Resolver
el sistema de ecuaciones:
    y - x^3 -2*x2 + 1 = 0,
    y + x^2 - 1 = 0,
el método de Newton-Raphson para sistemas de ecuaciones no lineales ya esta
implementado, consiste en la función optimize del módulo scipy
"""


def sistema(x):
    return [x[1] - x[0] ** 3 - 2 * x[0] ** 2 + 1, x[1] + x[0] ** 2 - 1]


def jacobiano_sistema(x):
    return [[- 3 * x[0] ** 2 - 4 * x[0], 1], [2 * x[0], 1]]


from scipy.optimize import fsolve  # Find the roots of a function.

x0 = np.array([[-2.5, -7], [-1, 2], [1, 2]], float)  # estimaciones iniciales

sol1 = fsolve(sistema, x0[0])
sol2 = fsolve(sistema, x0[1])
sol3 = fsolve(sistema, x0[2])

print('Soluciones del sistema de ecuaciones utilizando optimize.fsolve de scipy')
print('Solución 1 = ', sol1)
print('Solución 2 = ', sol2)
print('Solución 3 = ', sol3)

print('Solución del sistema de ecuaciones utilizando Newton-Raphson múltiples variables')

sol, it = newton_raphson_multiple_variables(sistema, jacobiano_sistema, x0, 1e-6)
print('Solución 1 = ', sol[0])
print('Solución 2 = ', sol[1])
print('Solución 3 = ', sol[2])
