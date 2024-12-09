import numpy as np
from gaussxw import *


# ----------------------------------------------Trapezium rule------------------------------------------------------------
def trapezium_rule(function, init_point, end_point, number_of_points):
    """
    :param function: Function for which to calculate the integral.
    :param init_point: Initial point of the integration interval.
    :param end_point: End point of the integration interval.
    :param number_of_points: Number of points to use in the integration.
    :return value: Value of the integral.
    """
    h = (end_point - init_point) / number_of_points
    value = (function(init_point) + function(end_point)) / 2
    for i in range(1, number_of_points):
        value += function(init_point + h * i)
    return value * h


def trapezium_rule_discrete(discrete_values, init_point, end_point):
    """
    :param discrete_values: Discrete values of the function for which to calculate the integral.
    :param init_point: Initial point of the integration interval.
    :param end_point: End point of the integration interval.
    :return value: Value of the integral.
    """
    number_of_points = len(discrete_values)
    h = (end_point - init_point) / (number_of_points - 1)
    value = (discrete_values[0] + discrete_values[-1]) / 2
    value += sum(discrete_values[1:-1])
    return value * h


# ----------------------------------------------Simpson rule-------------------------------------------------------------
def simpson_rule(function, init_point, end_point, number_of_points):
    """
    :param function: Function for which to calculate the integral.
    :param init_point: Initial point of the integration interval.
    :param end_point: End point of the integration interval.
    :param number_of_points: Number of points to use in the integration (must be odd).
    :return value: Value of the integral.
    """
    if number_of_points % 2 != 0:
        raise Exception("Number of points must be an odd number")
    h = (end_point - init_point) / number_of_points
    value = function(init_point) + function(end_point) + 4 * function(end_point - h)
    for k in range(1, number_of_points // 2):
        value += 4 * function(init_point + (2 * k - 1) * h) + 2 * function(init_point + 2 * k * h)
    return value * h / 3


def simpson_rule_discrete(discrete_values, init_point, end_point):
    """
    :param discrete_values: Discrete values of the function for which to calculate the integral.
    :param init_point: Initial point of the integration interval.
    :param end_point: End point of the integration interval.
    :return value: Value of the integral.
    """
    number_of_points = len(discrete_values) - 1
    if number_of_points % 2 != 0:
        raise Exception("Number of points must be an odd number")
    h = (end_point - init_point) / number_of_points
    value = discrete_values[0] + discrete_values[-1]
    for k in range(1, number_of_points - 1, 2):
        value += 4 * discrete_values[k]
    for k in range(2, number_of_points - 2, 2):
        value += 2 * discrete_values[k]
    return value * h / 3


# --------------------------------------------------------Adaptive methods-----------------------------------------------

def trapezium_rule_adaptive(function, init_point, end_point, eps_obj):
    """
    :param function: Function for which to calculate the integral.
    :param init_point: Initial point of the integration interval.
    :param end_point: End point of the integration interval.
    :param eps_obj: Error target for the adaptive method.
    :return:
        - value: Value of the integral.
        - it: Number of iterations performed.
        - n: Number of points used in the integration.
    """
    eps = 1.0
    n = 1
    h = (end_point - init_point) / n
    value1 = h / 2 * (function(init_point) + function(end_point))
    it = 1
    while eps > eps_obj:
        h /= 2
        value2 = 0
        for k in range(n):
            value2 += function(init_point + h * (2 * k + 1))
        value2 = value1 / 2 + h * value2
        eps = abs(value2 - value1) / 3
        it += 1
        value1 = value2
        n *= 2
    return value1, it, n


def romberg_rule(function, init_point, end_point, eps_obj):
    """
    :param function: Function for which to calculate the integral.
    :param init_point: Initial point of the integration interval.
    :param end_point: End point of the integration interval.
    :param eps_obj: Error target for the adaptive method.
    :return:
        - value: Value of the integral.
        - it: Number of iterations performed.
        - n: Number of points used in the integration.
    """
    eps = 1.0
    it = 1
    n = 1
    h = (end_point - init_point) / n
    value1 = h / 2 * (function(init_point) + function(end_point))
    r1 = np.array([value1], float)
    while eps > eps_obj:
        h /= 2
        value2 = 0
        for k in range(n):
            value2 += function(init_point + (2 * k + 1) * h)
        value2 = value1 / 2 + h * value2
        it += 1
        r2 = np.empty(it, float)
        r2[0] = value2
        for m in range(1, it):
            eps = abs(r2[m - 1] - r1[m - 1]) / (4 ** m - 1)
            r2[m] = r2[m - 1] + eps
        n *= 2
        value1 = value2
        r1 = r2
    return r1[-1], it, n


def simpson_rule_adaptive(function, init_point, end_point, eps_obj):
    """
    :param function: Function for which to calculate the integral.
    :param init_point: Initial point of the integration interval.
    :param end_point: End point of the integration interval.
    :param eps_obj: Error target for the adaptive method.
    :return:
        - value: Value of the integral.
        - it: Number of iterations performed.
        - n: Number of points used in the integration.
    """
    eps = 1
    it = 1
    n = 2
    h = (end_point - init_point) / n
    s = (function(init_point) + function(end_point)) / 3
    t = function(init_point + h) * 2 / 3
    value1 = h * (s + 2 * t)
    while eps > eps_obj:
        h /= 2
        s += t
        t = 0
        for k in range(n):
            t += 2 * function(init_point + (2 * k + 1) * h) / 3
        value2 = h * (s + 2 * t)
        eps = abs(value2 - value1) / 15
        it += 1
        value1 = value2
        n *= 2
    return value1, it, n


# --------------------------------------------------------Gaussian quadrature--------------------------------------------

def gaussian_quadrature(function, init_point, end_point, number_of_points):
    """
   :param function: Function for which to calculate the integral.
   :param init_point: Initial point of the integration interval.
   :param end_point: End point of the integration interval.
   :param number_of_points: Number of points to use in the integration.
   :return value: Value of the integral.
    """
    value = 0.0
    if init_point == -1 and end_point == 1:  # canonic interval
        x, w = gaussxw(number_of_points)
        for k in range(number_of_points):
            value += (w[k] * function(x[k]))
        return value
    else:
        x, w = gaussxwab(number_of_points, init_point, end_point)
        for k in range(number_of_points):
            value += (w[k] * function(x[k]))
        return value


"""
Chuleta:

1) Si la función es suficientemente suave los métodos de orden superior Romberg, Newton-Cotes
o la cuadratura Gaussiana permiten obtener gran precisión con pocos puntos.

2) Si la función no es suave, los métodos más sencillos, como el trapezoidal, suelen ser el camino
adecuado.

- Método Trapezoidal:
    • sencillo de programar (respuesta rápida aunque menos precisa).
    • usa puntos igualmente espaciados (para utilizar en datos de laboratorio).
    • útil para funciones con problemas (singularidades, ruido,. . . ).
    • en su forma de paso adaptado podemos garantizar una precisión a costa de más tiempo de computo.
    
- Método de Simpson:
    • sencillo de programar (respuesta rápida aunque menos precisa).
    • usa puntos equiespaciados (para utilizar en datos de laboratorio).
    • mayor precisión que el trapezoidal con los mismos puntos.
    • mayor orden de la aproximación del integrando (problemas si la función tiene ruido,. . . ).
    • en su forma de paso adaptado podemos garantizar una precisión, con posibles problemas si la función no es lo suficientemente suave.

- Método de Romberg:
    • con puntos igualmente espaciados es el mejor con órdenes altos de aproximación.
    • aproximación excepcional con pocos puntos.
    • garantiza una precisión con estimación del error.
    • problemas para funciones que se comporten mal, con ruido, singularidades,. . .
    • muy útil para funciones suaves cuya forma puede ser determinada por pocos puntos.
    
- Cuadratura Gaussiana:
    • puntos no equiespaciados.
    • similares ventajas e inconvenientes que el método de Romberg.
    • complejidad: cálculo de los puntos y los pesos.
    • mayor orden de aproximación.
"""
