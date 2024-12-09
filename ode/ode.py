import numpy as np
from intro.basics import count_arguments

# -----------------------------------------First order ODE--------------------------------------------------------------
"""
dx/dt = f(x,t) where f(x,t) could be linear or non-linear
"""


def euler(f, x0, a, b, n):
    """
    :param f: Function representing the ODE. Can be a function of two variables (x, t) or a single variable (x).
    :param x0: Initial condition, starting value of the dependent variable (x(t= 0) = x0)
    :param a: Initial point of the independent variable.
    :param b: Endpoint of the independent variable.
    :param n: Number of points to calculate between a and b.
    :return:
        - tp: Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
    """
    x = x0
    h = (b - a) / n
    tp = np.linspace(a, b, n)
    xp = []
    if count_arguments(f) == 2:
        for t in tp:
            xp.append(x)
            x += h * f(x, t)
    else:
        for t in tp:
            xp.append(x)
            x += h * f(x)
    return np.array(tp), np.array(xp)


def runge_kutta_2order(f, x0, a, b, n):
    """
    :param f: Function representing the ODE. Can be a function of two variables (x, t) or a single variable (x).
    :param x0: Initial condition, starting value of the dependent variable (x(t= 0) = x0)
    :param a: Initial point of the independent variable.
    :param b: Endpoint of the independent variable.
    :param n: Number of points to calculate between a and b.
    :return:
        - tp: Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
    """
    h = (b - a) / n
    x = x0
    tp = np.linspace(a, b, n)
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
    return np.array(tp), np.array(xp)


def runge_kutta_4order(f, x0, a, b, n):
    """
    :param f: Function representing the ODE. Can be a function of two variables (x, t) or a single variable (x).
    :param x0: Initial condition, starting value of the dependent variable (x(t= 0) = x0)
    :param a: Initial point of the independent variable.
    :param b: Endpoint of the independent variable.
    :param n: Number of points to calculate between a and b.
    :return:
        - tp: Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
    """
    h = (b - a) / n
    x = x0
    tp = np.linspace(a, b, n)
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
    return np.array(tp), np.array(xp)


def runge_kutta_4order_infinite_range(g, x0, n):
    """
    :param g: Function representing the ODE. Can be a function of two variables (x, t) or a single variable (x).
    :param x0: Initial condition, starting value of the dependent variable (x(t= 0) = x0)
    :param n: Number of points to calculate between a and b.
    :return:
        - tp: Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
    """
    a = 0
    b = 0.9999999999999999
    h = (b - a) / n
    x = x0
    up = np.linspace(a, b, n)
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
    return np.array(tp), np.array(xp)


# -----------------------------------------ODE systems-------------------------------------------------------------------
"""
dx/dt = f_x(x,t), dy/dt = f_y(x,t) where f_x(x,y,t) and f_y(x,y,t) could be linear or non-linear
"""


def runge_kutta_4order_system(system_function, r0, a, b, n):
    """
    :param system_function: The system of ODEs to solve.
    :param r0: Array of initial conditions for the system of ODEs.
    :param a: Initial point of the interval over which to integrate.
    :param b: End point of the interval over which to integrate.
    :param n: Number of points in the discretization of the interval [a, b].
    :return:
        - tp Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
        - yp: Numpy array of approximated values of the dependent variable.
    """
    h = (b - a) / n
    tp = np.linspace(a, b, n)
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
    return np.array(tp), np.array(xp), np.array(yp)


# ------------------------------------------Adaptive methods-------------------------------------------------------------

def runge_kutta_4order_system_adaptive(system_function, r0, tmax, delta, h0):
    """
    :param system_function: the system of ODEs to be solved, provided as an array
    :param r0: initial conditions of the system, given as an array
    :param tmax: the ending value of the independent variable t for which the solution is sought
    :param delta: the desired precision target used to adaptively adjust the step size
    :param h0: initial step size for the integration
    :return:
        - tp Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
        - yp: Numpy array of approximated values of the dependent variable.
    """
    r = np.copy(r0)
    xp = [r[0]]
    yp = [r[1]]
    t = 0
    tp = [0]
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
            tp.append(t)
            h *= min(rho ** 0.25, 2.0)
            r = r2
            xp.append(r[0])
            yp.append(r[1])
        else:
            h *= rho ** 0.25
    return np.array(tp), np.array(xp), np.array(yp)


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


def leapfrog(f, r0, a, b, h):
    """
    :param f: Function representing the ODE to solve, taking current state and time as arguments.
    :param r0: Initial condition of the dependent variable, usually a vector or scalar.
    :param a: Beginning of the interval over which to solve the ODE.
    :param b: End of the interval over which to solve the ODE.
    :param h: Step size for the integration, determining the interval's discretization granularity.
    :return:
        - tp: Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
    """
    tp = np.arange(a, b, h)
    r1 = np.copy(r0)
    xp = []
    r2 = r1 + h * f(r1) / 2
    if count_arguments(f) == 2:
        for t in tp:
            xp.append(r1[0])
            r1 += h * f(r2, t)
            r2 += h * f(r1, t)
    else:
        for t in tp:
            xp.append(r1[0])
            r1 += h * f(r2)
            r2 += h * f(r1)
    return np.array(tp), np.array(xp)


def verlet(f, kinetic_energy, potencial_energy, r0, v0, a, b, h):
    """
    :param f: Function representing the ODE to solve, taking current state and time as arguments.
    :param kinetic_energy: Function that calculates the kinetic.
    :param potencial_energy: Function to compute the potential energy.
    :param r0: Initial position of the object in the simulation.
    :param v0: Initial velocity of the object in the simulation.
    :param a: Start time for the simulation.
    :param b: End time for the simulation.
    :param h: Time step for the simulation.
    :return:
        - tp: Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
        - yp: Numpy array of approximated values of the dependent variable.
        - Vp: Numpy array of potential energies at each time point.
        - Tp: Numpy array of kinetic energies at each time point.
        - Ep: Numpy array of total energies at each time point.
    """
    r = np.copy(r0)
    v = np.copy(v0)
    vmitad = v + 0.5 * h * f(r)
    tp = np.arange(a, b, h)
    xp = []
    yp = []
    Vp = []
    Tp = []
    Ep = []
    if count_arguments(f) == 2:
        for t in tp:
            xp.append(r[0])
            yp.append(r[1])
            Vp.append(potencial_energy(r))
            Tp.append(kinetic_energy(v))
            Ep.append(potencial_energy(r) + kinetic_energy(v))
            r += h * vmitad
            k = h * f(r, t)
            v = vmitad + 0.5 * k
            vmitad += k
    else:
        for t in tp:
            xp.append(r[0])
            yp.append(r[1])
            Vp.append(potencial_energy(r))
            Tp.append(kinetic_energy(v))
            Ep.append(potencial_energy(r) + kinetic_energy(v))
            r += h * vmitad
            k = h * f(r)
            v = vmitad + 0.5 * k
            vmitad += k
    return tp, xp, yp, Vp, Tp, Ep


"""
Bulirsch-Stoer:
- Sin embargo, al ser un método basado en una serie perturbativa en h, las soluciones deben
tener un comportamiento suave para que la convergencia de la serie sea rápida.
- Por tanto, ecuaciones diferenciales con comportamientos patológicos (grandes fluctuaciones,
divergencias, ...) no son adecuados para este método.
- Cuando es aplicable, es considerado como uno de los mejores métodos para resolver ecuaciones diferenciales ordinarias.
"""


def bulirsch_stoer(f, r0, a, b, N, delta):
    """
    :param f: Function representing the ODE to solve, taking current state and time as arguments.
    :param r0: Initial conditions of the system at the start of the interval.
    :param a: Initial value of the independent variable (often time).
    :param b: Final value of the independent variable.
    :param N: Number of intervals into which the interval [a, b] is divided.
    :param delta: Desired accuracy per unit step length for the integration.
    :return:
        - tp: Numpy array of time points.
        - xp: Numpy array of approximated values of the dependent variable.
    """
    H = (b - a) / N
    r = np.copy(r0)
    tp = np.arange(a, b, H)
    xp = []
    if count_arguments(f) == 2:
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
    else:
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
    return np.array(tp), np.array(xp)


# ------------------------------------------Boundary conditions----------------------------------------------------------

def shoot(f, t_i, t_f, v1, v2, N, eps):
    """
    :param f: Function defining the system of differential equations to be solved.
    :param t_i: Initial time point of the integration.
    :param t_f: Final time point of the integration.
    :param v1: Initial guess for the initial condition at the start of integration.
    :param v2: Second guess for the initial condition at the start of integration, used in the bisection method.
    :param N: Number of time points in which the time interval [t_i, t_f] is divided.
    :param eps: Tolerance for convergence in the bisection method.
    :return: A tuple containing three elements:
        - tp: Numpy array of time points
        - xp: Numpy array of corresponding solution values.
        - v: The value of the initial condition that satisfies the boundary condition.
    """
    tp = np.linspace(t_i, t_f, N)
    h = (t_f - t_i) / N
    def g(v):
        r = np.array([0.0, v], float)
        for t in tp:
            k1 = h * f(r)
            k2 = h * f(r + 0.5 * k1)
            k3 = h * f(r + 0.5 * k2)
            k4 = h * f(r + k3)
            r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return r[0]
    h1 = g(v1)
    h2 = g(v2)
    while abs(h2 - h1) > eps:
        vp = (v1 + v2) / 2
        hp = g(vp)
        if h1 * hp > 0:
            v1 = vp
            h1 = hp
        else:
            v2 = vp
            h2 = hp
    v = (v1 + v2) / 2
    r = np.array([0.0, v], float)
    tp,xp,yp = runge_kutta_4order_system(f, r, t_i, t_f, N)
    return tp, xp, yp, v
