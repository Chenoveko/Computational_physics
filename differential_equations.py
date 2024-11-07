import numpy as np

from basics import count_arguments

# -----------------------------------------First order ODE--------------------------------------------------------------
"""
dx/dt = f(x,t) where f(x,t) could be linear or non-linear
"""


def euler_method(f: 'f(x,t)', x0: 'initial condition', a: 'initial point', b: 'end point',
                 N: 'Number of points') -> 'ODE solution':
    x = x0
    h = (b - a) / N
    tp = np.linspace(a, b, N)
    xp = []
    if count_arguments(f) == 2:
        for t in tp:
            xp.append(x)
            x += h * f(x, t)
    else:
        for t in tp:
            xp.append(x)
            x += h * f(x)
    return tp, np.array(xp)


def runge_kutta_2_method(f: 'f(x,t)', x0: 'initial condition', a: 'initial point', b: 'end point',
                         N: 'Number of points') -> 'ODE solution':
    h = (b - a) / N
    x = x0
    tp = np.linspace(a, b, N)
    xp = []
    if count_arguments(f) == 2:
        for t in tp:
            xp.append(x)
            k1 = h * f(x, t)
            k2 = h * f(x + k1 / 2, t + h / 2)
            x += k2
    else:
        for t in tp:
            xp.append(x)
            k1 = h * f(x)
            k2 = h * f(x + k1 / 2)
            x += k2
    return tp, np.array(xp)


def runge_kutta_4_method(f: 'f(x,t)', x0: 'initial condition', a: 'initial point', b: 'end point',
                         N: 'Number of points') -> 'ODE solution':
    h = (b - a) / N
    x = x0
    tp = np.linspace(a, b, N)
    xp = []
    if count_arguments(f) == 2:
        for t in tp:
            xp.append(x)
            k1 = h * f(x, t)
            k2 = h * f(x + k1 / 2, t + h / 2)
            k3 = h * f(x + k2 / 2, t + h / 2)
            k4 = h * f(x + k3, t + h)
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        for t in tp:
            xp.append(x)
            k1 = h * f(x)
            k2 = h * f(x + k1 / 2)
            k3 = h * f(x + k2 / 2)
            k4 = h * f(x + k3)
            x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return tp, np.array(xp)


def runge_kutta_4_infinite_range_method(g: 'g(x,u)', x0: 'initial condition', a: 'initial point', b: 'end point',
                                        N: 'Number of points') -> 'ODE solution':
    h = (b - a) / N
    x = x0
    up = np.linspace(a, b, N)
    tp = []
    xp = []
    for u in up:
        tp.append(u / (1 - u))
        xp.append(x)
        k1 = h * g(x, u)
        k2 = h * g(x + k1 / 2, u + h / 2)
        k3 = h * g(x + k2 / 2, u + h / 2)
        k4 = h * g(x + k3, u + h)
        x += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return tp, np.array(xp)


# -----------------------------------------ODE systems-------------------------------------------------------------------
"""
dx/dt = f_x(x,t), dy/dt = f_y(x,t) where f_x(x,y,t) and f_y(x,y,t) could be linear or non-linear
"""


def runge_kutta_4_system_method(system_function: 'system function as array', r0: 'initial conditions array',
                                a: 'initial point', b: 'end point',
                                N: 'Number of points') -> 'Solution of the ODE system':
    h = (b - a) / N
    tp = np.linspace(a, b, N)
    xp = []
    yp = []
    r = r0
    if count_arguments(system_function) == 2:
        for t in tp:
            xp.append(r[0])
            yp.append(r[1])
            k1 = h * system_function(r, t)
            k2 = h * system_function(r + 0.5 * k1, t + 0.5 * h)
            k3 = h * system_function(r + 0.5 * k2, t + 0.5 * h)
            k4 = h * system_function(r + k3, t + h)
            r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    else:
        for t in tp:
            xp.append(r[0])
            yp.append(r[1])
            k1 = h * system_function(r)
            k2 = h * system_function(r + 0.5 * k1)
            k3 = h * system_function(r + 0.5 * k2)
            k4 = h * system_function(r + k3)
            r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return tp, np.array(xp), np.array(yp)

# -----------------------------------------Second order ODE-------------------------------------------------------------