import numpy as np
#---------------------------------------------Metodo del pto fijo o relajación------------------------------------------
"""
    Se aplica a ecuaciones del tipo: x = f(x)
"""

def fixed_point_method(f,x0,eps):
    it = 0
    err = 1.0
    x0 = 1.0
    while err > eps:
        x1 = f(x0)
        err = abs(x1 - x0)
        it += 1
        x0 = x1
        print(it, x1)
    return x1, it

def newton_raphson_method(f:'Funcion',fp:'Derivada funcion',x1:'Estimación inicial',eps:'Precisión deseada') -> 'Raiz':
    """
    Más rápido que el metodo del punto fijo y bisección. Desventajas:
        - Necesitamos conocer la derivada de f(x).
        - El metodo no siempre converge:por ejemplo si f'(x) es muy pequeña, el metodo iterativo
          hará que el error de cada iteración sea mayor que en la anterior.
        - Puede ocurrir que la pendiente de la función apunte en el sentido contrario de la raíz.
    """
    it = 0
    while f(x1) > eps:
        x2 = x1 - f(x1) / fp(x1)
        x1 = x2
        it += 1
        print(it,x1)
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

def secante_method(f,x1,x2,eps):
    """
    Como el metodo de Newton-Raphson, pero no se conoce una expresión analítica de la derivada -> Se estima numericamente
    Mismas limitaciones qe Newton-Raphson
    """
    it = 0
    while f(x1) > eps:
        f_prima = (f(x2) - f(x1))/(x2-x1)
        x3 = x2 -
        x2 = x1 - f(x1) / fp(x1)
        x1 = x2
        it += 1
        print(it,x1)
    return x1,it



