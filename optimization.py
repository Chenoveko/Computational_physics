import numpy as np

from numpy.linalg import solve

"""
- Si es posible calcular la derivada:
    • Buscar los puntos estacionarios en los que se anulen la derivada o las derivadas parciales.
      Si estamos buscando el mínimo de una función: f (x1, x2, ...) resolvemos el sistema:
    
        df/dx_j = 0 ; j = 1,2...
    • Si el sistema de ecuaciones es lineal aplicamos la factorización LU o los métodos
      iterativos que hemos visto en la sección 5.1.
    • Si es un sistema no-lineal podemos utilizar el método del punto fijo/relajación o el método
      de Newton-Raphson. La aplicación del método de Newton-Raphson a la búsqueda de mínimos se llama método
      de Gauss-Newton.
- Si no es posible calcular la derivada, es necesario buscar el mínimo con métodos numéricos.
"""
#----------------------------------------Metodo de la razón aurea-------------------------------------------------------

def razon_aurea_minimizar(f:'funcion',x1:'pto inicial',x4:'pto final',eps:'Precisión objetivo'):
    z = (1 + np.sqrt(5)) / 2  # razón áurea
    x2 = x4 - (x4 - x1) / z
    x3 = x1 + (x4 - x1) / z
    f2 = f(x2)
    f3 = f(x3)
    while x4 - x1 > eps:
        if f2 < f3:
            x4, f4 = x3, f3
            x3, f3 = x2, f2
            x2 = x4 - (x4 - x1) / z
            f2 = f(x2)
        else:
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            x3 = x1 + (x4 - x1) / z
            f3 = f(x3)
    xmin = (x1 + x4) / 2
    return xmin

def razon_aurea_maximizar(f:'funcion',x1:'pto inicial',x4:'pto final',eps:'Precisión objetivo'):
    z = (1 + np.sqrt(5)) / 2  # razón áurea
    x2 = x4 - (x4 - x1) / z
    x3 = x1 + (x4 - x1) / z
    f2 = f(x2)
    f3 = f(x3)
    while x4 - x1 > eps:
        if f2 > f3:
            x4, f4 = x3, f3
            x3, f3 = x2, f2
            x2 = x4 - (x4 - x1) / z
            f2 = f(x2)
        else:
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            x3 = x1 + (x4 - x1) / z
            f3 = f(x3)
    xmin = (x1 + x4) / 2
    return xmin

def gauss_newton(J:'Jacobian',J2:'jacobian2',x0:'Estimación inicial',eps:'Precisión deseada',max_iter = 1000) -> 'Raiz':
    x1 = np.copy(x0)
    it = 0
    for i in x1:
        err = 1
        while err > eps:
            print(J2(i))
            print(J(i))
            dx = solve(J2(i), J(i))
            i -= dx
            err = abs(max(dx))
        x1[it] = i
        it += 1
    return x1,it