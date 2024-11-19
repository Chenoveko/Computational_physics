import numpy as np
from numpy.linalg import solve, eigvals
#---------------------------------------------Metodo del pto fijo o relajación------------------------------------------
"""
- Se aplica a ecuaciones del tipo: x = f(x)
- Si la derivada de la función en la solución cumple |f'(x)| > 1 el metodo no convergerá hacía la solución.
"""

def fixed_point(f,x0,eps, max_iter = 1000):
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

def fixed_point_multiple_variables(system,initial_estimation,eps):
    x0 = np.copy(initial_estimation)
    x1 = np.zeros(len(x0))
    for i in range(len(x0)):
        err = 1.0
        while err > eps:
            value = system(x0[i])
            x1[i] = system(x0[i])
            err = abs(x1[i] - x0[i])
            x0[i] = x1[i]
    return x1

#--------------------------Metodo del pto fijo o relajación con más de una variable-------------------------------------

#---------------------------------------------Metodo de la biseccion----------------------------------------------------
"""
- Más robusto que metodo del pto fijo
- Consiste en encontrar una solución de la ecuación: f (x) = 0, el metodo se basa por tanto en encontrar las raíces
de una ecuación
"""

def bisection_method(f:'funcion',a:'intervalo',b:'intervalo',eps:'tolerancia', max_iter = 1000):
    fa, fb = f(a), f(b)
    it = 0
    while b-a > eps:
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
    return m,it

#---------------------------------------------Metodo de Newton-Raphson--------------------------------------------------
"""
Más rápido que el metodo del punto fijo y bisección. Desventajas:
    - Necesitamos conocer la derivada de f(x).
    - El metodo no siempre converge:por ejemplo si f'(x) es muy pequeña, el metodo iterativo
      hará que el error de cada iteración sea mayor que en la anterior.
    - Puede ocurrir que la pendiente de la función apunte en el sentido contrario de la raíz.
"""
def newton_raphson_method(f:'Funcion',fp:'Derivada funcion',x1:'Estimación inicial',eps:'Precisión deseada',max_iter = 1000) -> 'Raiz':
    it = 0
    while f(x1) > eps:
        x2 = x1 - f(x1) / fp(x1)
        x1 = x2
        it += 1
        print(it,x1)
        if it >= max_iter:
            print("Se ha alcanzado el número máximo de iteraciones en el método de Newton-Raphson")
            break
    return x1,it


def newton_raphson_multiple_variables(f:'Funcion',J:'jacobian',x0:'Estimación inicial',eps:'Precisión deseada',max_iter = 1000) -> 'Raiz':
    x1 = np.copy(x0)
    it = 0
    for i in x1:
        err = 1
        while err > eps:
            dx = solve(J(i), f(i))
            i -= dx
            err = abs(max(dx))
        x1[it] = i
        it += 1
    return x1,it

"""
Newton-Raphson en varias variables --> Consiste en la función optimize del módulo scipy
from scipy import optimize
sol1=optimize.fsolve(f,x0[0])
sol2=optimize.fsolve(f,x0[1])
sol3=optimize.fsolve(f,x0[2])
print(sol1,x1[0])
print(sol2,x1[1])
print(sol3,x1[2])
"""

"""
def secante_method(f,x1,x2,eps):
    
    Como el metodo de Newton-Raphson, pero no se conoce una expresión analítica de la derivada -> Se estima numericamente
    Mismas limitaciones qe Newton-Raphson
    
    it = 0
    while f(x1) > eps:
        f_prima = (f(x2) - f(x1))/(x2-x1)
        x3 = x2 -
        x2 = x1 - f(x1) / fp(x1)
        x1 = x2
        it += 1
        print(it,x1)
    return x1,it
"""

