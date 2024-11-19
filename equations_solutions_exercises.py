from eigenvalues_eigenvectors import *
from linear_equations_solutions import *
from non_linear_equations import *
import matplotlib.pyplot as plt

from optimization import razon_aurea

#------------------------------Linear equations solutions--------------------------------------------------------------
""""
Ejemplo 5.1: Eliminación gaussiana
Usar el metodo de la eliminación de Gauss para transformar el sistema de ecuaciones:
    2w + x + 4 y + z = - 4,
    3w + 4 x - y - z =3,
    w - 4 x + y + 5 z =9,
    2w - 2 x + y + 3 z =7.
"""
# Definimos nuestra matriz ampliada
A=np.array([[2,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
B=np.array([-4,3,9,7],float)

x, AB2 = gaussian_elimination(A,B)
AB=np.column_stack((A,B))
print("Nuestra matriz extendida inicial")
print(AB)
print("")
print("Nuestra matriz extendida triangular")
print(AB2)
print("")
print("La solución del sistema de ecuaciones es:")
print(x)
print("")
print("Comprobamos que es solución")
print(np.dot(A,x)-B)

""""
Ejemplo 5.2: Eliminación gaussiana incluyendo el pivote
Resolver el sistema incluyendo el pivote

    x + 4 y + z = - 4,
    3w + 4 x - y - z =3,
    w - 4 x + y + 5 z =9,
    2w - 2 x + y + 3 z =7.

"""
# Definimos nuestra matriz ampliada
A=np.array([[0,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
B=np.array([-4,3,9,7],float)

x, AB2 = gaussian_elimination_pivot(A,B)

AB=np.column_stack((A,B))
print("Nuestra matriz extendida inicial")
print(AB)
print("")
print("Nuestra matriz extendida triangular")
print(AB2)
print("")
print("La solución del sistema de ecuaciones es:")
print(x)
print("")
print("Comprobamos que es solución")
print(np.dot(A,x)-B)

""""
Ejemplo 5.5: Aplicando la descomposición LU 
"""
A=np.array([[2,1,4,1],[3,4,-1,-1],[1,-4,1,5],[2,-2,1,3]],float)
V=np.array([-4,3,9,7],float)
# Generamos las matrices L y U y comprobamos que son como deberían ser.
L,U=LU_factorization(A)
print(L)
print()
print(U)
print()
# Comprobamos que L U es efectivamente A.
print(np.dot(L,U))
print()
print(A)
# Solución del sistema usando LU
print("Solución del sistema usando LU",LU_solution(A,V))
#--------------------------------------Eigenvalues and eigenvectors-----------------------------------------------------
"""
Ejemplo 5.12: estudiando la descomposición QR
"""
A=np.array([[1,4,8,4],
[4,2,3,7],
[8,3,6,9],
[4,7,9,2]],float)

Q,R=QR_factorization(A)
print("Q = ",Q)
print()
print("R = ",R)
print()
print(np.dot(Q,R)-A)

"""
Ejemplo 5.13: calculando autovalores y autovectores
Escribir un programa que utilice la factorización QR para obtener los autovalores de A asegurando
que los elementos de fuera de la diagonal son menores de 10􀀀6.
"""
A=np.array([[1,4,8,4],
[4,2,3,7],
[8,3,6,9],
[4,7,9,2]],float)

D,V=QR_diagonalization(A,1e-12)

print("D = ",D)
print()
print("V = ",V)
print()
print("Nuestros autovalores son: ",np.diag(D))

# Comprobemos que son autovalores y autovectores:
for i in range(len(A)):
    print(np.dot(A,V[:,i])- np.diag(D)[i]*V[:,i])

#-----------------------------------------------------Non linear equations----------------------------------------------
"""
Ejemplo 5.14: aplicando el método del punto fijo
Aplicar el método del punto fijo para encontrar la solución a la ecuación
    x = 2 - e^-x
que no tiene solución analítica
"""
def f(x):
    return 2-np.exp(-x)

x0 = 1.0
eps=1e-6
print('Ejemplo 5.14')
fixed_point_method(f,x0,eps)

"""
Ejemplo 5.15: aplicando el método del punto fijo por segunda vez aplicar el método del punto
fijo para encontrar la solución a la ecuación
    x = e^(1-x^2)
"""

def g(x):
    return np.exp(1-x**2)

x0 = 0
print('Ejemplo 5.15')
m,it = fixed_point_method(g,x0,eps)

print('Punto fijo en x = ',m)
print('Iteraciones método pto fijo = ',it)


"""
Ejemplo 5.16: aplicando el método del punto fijo por tercera vez
    x = x^2 + sin(2x) -> x = arcsin(x-x^2) * 1/2
"""

def g516(x):
    return np.arcsin(x - x * x) / 2

x0 = 1e-4
fixed_point_method(g516,x0,eps)

""""
Ejemplo 5.17: visualizando el metodo de la biseccion. Encontrar la raíz de la función:
    f(x) = e^x - 2
"""

def f517(x):
    return np.exp(x)-2

# En el intervalo [-2,2] tenemos seguro una raíz.
a,b = -2,2
eps=1e-6
m,it = bisection_method(f517,a,b,eps)
print('Raíz en x = ',m)
print('Iteraciones del método bisección = ',it)

""""
Ejemplo 5.18: visualizando el metodo de la Newton-Raphson. Encontrar la raíz de la función:
    f(x) = e^x - 2
"""
def fp517(x):
    return np.exp(x)

x1 = 2
raiz,it = newton_raphson_method(f517,fp517,x1,eps)
print('Raíz en x = ',raiz)
print('Iteraciones del método Newton-Raphson = ',it)

"""
Ejemplo 5.19: representación gráfica del método de Newton-Raphson en varias variables. Resolver
el sistema de ecuaciones:
    y - x^3 -2*x2 + 1 = 0,
    y + x^2 - 1 = 0,
el método de Newton-Raphson para sistemas de ecuaciones no lineales ya esta
implementado, consiste en la función optimize del módulo scipy
"""

def sistema_ecuaciones(x):
    return [x[1] - x[0]**3 - 2 * x[0]**2 + 1, x[1] + x[0]**2 - 1]

from scipy import optimize

x0=np.array([[-2.5, -7],[-1, 2],[1, 2]],float) # estimaciones iniciales


sol1=optimize.fsolve(sistema_ecuaciones,x0[0])
sol2=optimize.fsolve(sistema_ecuaciones,x0[1])
sol3=optimize.fsolve(sistema_ecuaciones,x0[2])

print('Soluciones del sistema de ecuaciones utilizando optimize de scipy')
print('Solución 1 = ',sol1)
print('Solución 2 = ',sol2)
print('Solución 3 = ',sol3)
#----------------------------------------------------Optimization-------------------------------------------------------
"""
Ejemplo 5.20: el potencial de Buckingham
El potencial de Buckingham es una representación aproximada de la energía de interacción entre
los átomos de un sólido ó gas como función de la distancia r entre ellos
"""

def potencial_buckingham(r,sigma =1):
    return (sigma/r)**6-np.exp(-r/sigma)

# Lo representamos para estimar el intervalo
x = np.linspace(0.1,10,1000)
plt.ylim(-0.2,0.1)
plt.title('Potencial de Buckingham')
plt.plot(x,potencial_buckingham(x))
plt.show()

x1= 0.1
x4= 10
eps=1e-6
xmin = razon_aurea(potencial_buckingham,x1,x4,eps)
print("El mínimo cae en",xmin,"nm")

plt.plot(x,potencial_buckingham(x))
plt.title('Potencial de Buckingham con mínimo')
plt.plot(xmin,potencial_buckingham(xmin),'ko')
plt.ylim(-0.2,0.1)
plt.show()


