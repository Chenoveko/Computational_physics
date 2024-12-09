import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------Fixed point method--------------------------------------------------
"""
- Se aplica a ecuaciones del tipo: x = f(x)
- Si la derivada de la función en la solución cumple |f'(x)| > 1 el metodo no convergerá hacía la solución.
"""


def fixed_point(f, x0, eps, max_iter=1000):
    """
    :param f: Function for which the fixed point is to be found.
    :param x0: Initial guess for the fixed point.
    :param eps: Desired accuracy for the convergence criterion.
    :param max_iter: Maximum number of iterations allowed (default is 1000).
    :return:
        - x1: Estimated fixed point and the number of iterations performed.
        - it: Number of iterations taken to reach the specified precision.
    """
    it = 0
    err = 1.0
    while err > eps:
        x1 = f(x0)
        err = abs(x1 - x0)
        it += 1
        x0 = x1
        if it >= max_iter:
            print("Se ha alcanzado el número máximo de iteraciones en el método del pto fijo")
            break
    return x1, it


def fixed_point_multiple_variables(system, initial_estimation, eps):
    """
    :param system: A function representing the system of equations to solve.
    :param initial_estimation: An initial guess for the solutions.
    :param eps: Desired accuracy for the convergence criterion.
    :return x1: A tuple containing the solutions for the system of equations.
    """
    x0 = np.copy(initial_estimation)
    pos = 0
    for i in x0:
        err = 1
        while err > eps:
            i2 = np.array(system(i))
            err = max(abs(i2 - i))
            i = i2
        x0[pos] = i
        pos += 1
    return x0


def fixed_point_multiple_variables_no_convergente(system, initial_estimation):
    """
    :param system: A function representing the system of equations to solve.
    :param initial_estimation: An initial guess for the solutions.
    :return x1: A tuple containing the solutions for the system of equations.
    """
    x0 = np.copy(initial_estimation)
    pos = 0
    for i in x0:
        for k in range(10):
            i2 = np.array(system(i))
            i = i2
        x0[pos] = i
        pos += 1
    return x0


def fixed_point_2_variables(system, x0, eps, max_iter=10):
    """
        :param system: A function representing the system of equations to solve.
        :param x0: An initial guess for the solutions.
        :param eps: Desired accuracy for the convergence criterion.
        :param max_iter: Maximum number of iterations allowed (default is 1000).
        :return x2: A tuple containing the solutions for the system of equations.
        """
    x2 = np.copy(x0)
    pos = 0
    for i in x2:
        err = 1
        it = 0
        while err > eps:
            i2 = np.array(system(i))
            err = max(abs(i2 - i))
            i = i2
            it += 1
            if it >= max_iter:
                print('Se ha alcanzado el número máximo de iteraciones')
                break
        x2[pos] = i
        pos += 1
    plt.plot(x2[:, 0], x2[:, 1], 'b*')
    return x2


# ---------------------------------------------Bisection method----------------------------------------------------------

def bisection(f, a, b, eps, max_iter=100):
    """
    :param f: Function for which the root is being sought.
    :param a: Lower bound of the interval.
    :param b: Upper bound of the interval.
    :param eps: Tolerance for stopping criterion.
    :param max_iter: Maximum number of iterations allowed (default is 100).
    :return:
        - m: Estimated root.
        - it: Number of iterations performed.
    """
    fa, fb = f(a), f(b)
    it = 0
    while b - a > eps:
        m = a + (b - a) / 2
        fm = f(m)
        it += 1
        if np.sign(fa) == np.sign(fm):
            a, fa = m, fm
        else:
            b, fb = m, fm
        if it >= max_iter:
            print("Se ha alcanzado el número máximo de iteraciones en el método de la bisección")
            break
    return m, it


# ---------------------------------------------Newton-Raphson method-----------------------------------------------------
"""
Más rápido que el metodo del punto fijo y bisección. Desventajas:
    - Necesitamos conocer la derivada de f(x).
    - El metodo no siempre converge:por ejemplo si f'(x) es muy pequeña, el metodo iterativo
      hará que el error de cada iteración sea mayor que en la anterior.
    - Puede ocurrir que la pendiente de la función apunte en el sentido contrario de la raíz.
"""


def newton_raphson(f, fp, x1, eps, max_iter=100):
    """
    :param f: Function for which the root is being sought.
    :param fp: Derivative of the function f(x).
    :param x1: Initial estimation for the root.
    :param eps: Tolerance for stopping criterion.
    :param max_iter: Maximum number of iterations allowed (default is 100).
    :return:
        - m: Estimated root.
        - it: Number of iterations performed.
    """
    it = 0
    while f(x1) > eps:
        x2 = x1 - f(x1) / fp(x1)
        x1 = x2
        it += 1
        if it >= max_iter:
            print("Se ha alcanzado el número máximo de iteraciones en el método de Newton-Raphson")
            break
    return x1, it


def newton_raphson_multiple_variables(F, JF, x0, eps):
    """
    :param F: A function representing the system of nonlinear equations to be solved.
    :param JF: A function representing the Jacobian matrix of the system.
    :param x0: A numpy array representing the initial guess for the variables of the system.
    :param eps: Tolerance for stopping criterion.
    :return:
        - x1: A tuple containing the solutions for the system of equations.
        - it: Number of iterations performed.
    """
    x1 = np.copy(x0)
    it = 0
    for i in x1:
        err = 1
        while err > eps:
            dx = np.linalg.solve(JF(i), F(i))
            i -= dx
            err = abs(max(dx))
        x1[it] = i
        it += 1
    return x1, it


def secant(f, a, b, eps, max_iter=100):
    """
    :param f: Function for which the root is being sought.
    :param a: Lower bound of the interval.
    :param b: Upper bound of the interval.
    :param eps: Tolerance for stopping criterion.
    :param max_iter: Maximum number of iterations allowed (default is 100).
    :return:
        - m: Estimated root.
        - it: Number of iterations performed.
    """
    it = 0
    x0, x1 = a, b
    while abs(f(x1)) > eps and it < max_iter:
        try:
            # Calcula la derivada de la secante
            f_prima = (f(x1) - f(x0)) / (x1 - x0)
            # Nuevo punto
            x2 = x1 - f(x1) / f_prima
        except ZeroDivisionError:
            print("Error: división por cero. El método de la secante falla.")
            return None, it
        x0, x1 = x1, x2
        it += 1
    return x1, it
