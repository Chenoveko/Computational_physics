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
    r = np.copy(r0)
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


# -----------------------------------------Adaptive methods--------------------------------------------------------------

def runge_kutta_4_system_adaptive_method(system_function: 'system function as array', r0: 'initial conditions array',
                                         tmax: 'end point', delta: 'precision objetivo',
                                         h0: 'paso inicial') -> 'Solution of the ODE system':
    r = np.copy(r0)
    xp = [r[0]]
    yp = [r[1]]
    t = 0
    h = h0
    while t < tmax:
        # hacemos un paso grande
        k1 = 2 * h * system_function(r)
        k2 = 2 * h * system_function(r + k1 / 2)
        k3 = 2 * h * system_function(r + k2 / 2)
        k4 = 2 * h * system_function(r + k3)
        r1 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # hacemos dos pasos pequeños
        k1 = h * system_function(r)
        k2 = h * system_function(r + k1 / 2)
        k3 = h * system_function(r + k2 / 2)
        k4 = h * system_function(r + k3)
        r2 = r + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        k1 = h * system_function(r2)
        k2 = h * system_function(r2 + k1 / 2)
        k3 = h * system_function(r2 + k2 / 2)
        k4 = h * system_function(r2 + k3)
        r2 += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        # calculamos el valor del ratio de precisiones rho
        dx = r1[0] - r2[0]  # error en x
        dy = r1[1] - r2[1]  # error en y
        rho = 30 * h * delta / np.sqrt(dx ** 2 + dy ** 2)
        # calculamos el nuevo valor de t,h y r
        if rho > 1:
            t += 2 * h
            h *= min(rho ** 0.25, 2.0)
            r = r2
            xp.append(r[0])
            yp.append(r[1])
        else:
            h *= rho ** 0.25
    return xp, yp

# -----------------------------------------Other methods----------------------------------------------------------------
"""
Método del salto de rana:
- al cabo de un ciclo completo la energía del sistema calculada con el método
del salto de la rana debe volver al mismo valor y seguiría así para siempre, cosa que no
sucede con el método de Runge-Kutta
- el método del salto de la rana puede ser muy útil para para estudiar sistemas que
conservan la energía en periodos de tiempo grandes. Mientras que para un Runge-Kutta
un sistema planetario o un péndulo podrían acabar parándose, la solución obtenida con el
método del salto de la rana podría durar siempre.
"""
def leapfrog_method(f: 'f(x,t)', r1: 'initial condition', a: 'initial point', b: 'end point',
                         h: 'paso') -> 'ODE solution':
    # discretizamos nuestro intervalo
    tp = np.arange(a, b, h)
    xp = []
    r2 = r1 + h * f(r1) / 2
    # loop principal
    for t in tp:
        xp.append(r1[0])
        r1 += h * f(r2)
        r2 += h * f(r1)
    return tp,xp

def verlet_method(f: 'f(x,t)',T:'Energia cinetica',V:'Energia potencial', r0: 'initial condition',v0:'initial condition', a: 'initial point', b: 'end point',
                         h: 'paso') -> 'ODE solution':
    r = np.copy(r0)
    v = np.copy(v0)
    vmitad = v + 0.5 * h * f(r)
    tp = np.arange(a, b, h)
    xp = []
    yp = []
    Vp = []
    Tp = []
    Ep = []
    # bucle principal
    for t in tp:
        xp.append(r[0])
        yp.append(r[1])
        Vp.append(V(r))
        Tp.append(T(v))
        Ep.append(V(r) + T(v))
        r += h * vmitad
        k = h * f(r)
        v = vmitad + 0.5 * k
        vmitad += k
    return tp,xp,yp,Vp,Tp,Ep


def bulirsch_Stoer_4(f: 'f(x,t)', r0: 'initial condition', a: 'initial point', b: 'end point',
                         N: 'Number of points',delta : 'precision objetivo') -> 'ODE solution':
    H = (b - a) / N
    r = np.copy(r0)
    tp = np.linspace(a, b, N)
    xp = []
    for t in tp:
        xp.append(r[0])
        n = 1
        r1 = r + 0.5 * H * f(r)
        r2 = r + H * f(r1)
        R1 = np.empty([n, 2], float)
        R1[0] = 0.5 * (r1 + r2 + 0.5 * H * f(r2))
        error = 2 * H * delta
        while error > H * delta:
            n += 1
            h = H / n
            r1 = r + 0.5 * h * f(r)
            r2 = r + h * f(r1)
            for i in range(n - 1):
                r1 += h * f(r2)
                r2 += h * f(r1)
            R2 = R1  
            R1 = np.empty([n, 2], float)
            R1[0] = 0.5 * (r1 + r2 + 0.5 * h * f(r2))
            for m in range(1, n):
                eps = (R1[m - 1] - R2[m - 1]) / ((n / (n - 1)) ** (2 * m) - 1)
                R1[m] = R1[m - 1] + eps
            error = abs(eps[0])
            r = R1[n - 1]
    return tp, np.array(xp)


