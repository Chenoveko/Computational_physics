import numpy as np
import matplotlib.pyplot as plt
from differential_equations import *

# -----------------------------------------First order ODE--------------------------------------------------------------
"""
Ejemplo 7.1: aplicando el metodo de Euler en el intervalo [0, 10] usando 1000 puntos equiespaciados 
con la condición inicial x(t = 0) = 0

    dx/dt =-x^3 + sin(t)
"""


def f(x, t):
    return -x ** 3 + np.sin(t)


a = 0  # comienzo del intervalo
b = 10  # final del intervalo
N = 1000  # número de pasos
x0 = 0  # condición inicial

tp, xp = euler_method(f, x0, a, b, N)
plt.plot(tp, xp)
plt.title("Euler method")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

"""
Ejemplo 7.2: aplicando el metodo de Runge-Kutta de orden dos en el intervalo [0, 10] usando 100 puntos equiespaciados 
con la condición inicial x(t = 0) = 0
Compararlo con el resultado obtenido con el método de Euler para 100 pasos.

    dx/dt =-x^3 + sin(t)
"""

tpp, xpp = runge_kutta_2_method(f, x0, a, b, N)
plt.plot(tp, xp, label="Euler")
plt.plot(tpp, xpp, label="Runge-Kutta second order")
plt.legend()
plt.title("Euler Method and Runge-Kutta 2 method")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

"""
Ejemplo 7.3: aplicando el metodo de Runge-Kutta de cuarto orden en el intervalo [0, 10] usando 50 puntos equiespaciados 
con la condición inicial x(t = 0) = 0
Compararlo con el método de Euler y el método de Runge-Kutta de orden dos para N = 100.

    dx/dt =-x^3 + sin(t)
"""

tppp, xppp = runge_kutta_4_method(f, x0, a, b, 50)
plt.plot(tp, xp, label="Euler")
plt.plot(tpp, xpp, label="Runge-Kutta second order")
plt.plot(tppp, xppp, 'o', label="Runge-Kutta forth order")
plt.legend()
plt.title("Euler Method and Runge-Kutta 2,4 method")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()

"""
Ejemplo 7.4 Ecuaciones diferenciales en rangos infinitos: usar el método de Runge-Kutta de cuarto orden para obtener
 su solución en el intervalo t 2 [0,inf) con la condición inicial x(0)=1. Representarla en el intervalo [0,100].

    dx/dt = 1/(x^2+t^2)
    
Haciendo el cambio de variable t = u/(1-u) tenemos:

    dx/dt = 1/[(1-u)^2*x^2+u^2] = g(x,u), con u perteneciente a [0, 1].
"""


def g(x, u):
    return 1 / (x ** 2 * (1 - u) ** 2 + u ** 2)


a = 0.0  # límite inferior en u
b = 0.99999  # límite superior en u
N = 100  # número de puntos
x0 = 1  # condición inicial

tp, xp = runge_kutta_4_infinite_range_method(g, x0, a, b, N)
plt.plot(tp, xp)
plt.xlim(1, 100)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.show()
# -----------------------------------------ODE systems-------------------------------------------------------------------
"""
Ejemplo 7.5: Sistemas de ecuaciones diferenciales ordinarias: encontrar su solución en el intervalo t e [0, 10], 
con x(0)=y(0)=1.

    dx/dt = xy - x
    dy/dt = y -xy + sin(t)^2
"""


def ode_system(r, t):
    x = r[0]
    y = r[1]
    fx = x * y - x
    fy = y - x * y + np.sin(t) ** 2
    return np.array([fx, fy], float)


a = 0  # punto inicial del intervalo
b = 10  # punto final del intervalo
r0 = np.array([1, 1], float)  # condiciones iniciales
N = 1000  # número de puntos

tp, xp, yp = runge_kutta_4_system_method(ode_system, r0, a, b, N)
plt.plot(tp, xp)
plt.plot(tp, yp)
plt.title("Ode system")
plt.xlabel("t")
plt.show()

"""
Ejercicio 7.2: las ecuaciones de Lotka-Volterra
Escribir un programa que resuelva las ecuaciones usando el método de Runge-Kutta de
cuarto orden con alfa = 1, beta = gamma = 0,5 y delta = 2, usando las condiciones iniciales x(0)=y(0)=2 en
unidades de miles de individuos y representar sus poblaciones en el intervalo t e [0, 30].

    Población de conejos -> x , Población de zorros -> y
    dx/dt = alfa*x - beta*x*y
    dy/dt = gamma*x*y - delta*y
    todas las constantes son positivas
"""


def lokta_volterra(r):
    alfa, beta, gamma, delta = 1, 0.5, 0.5, 2
    x = r[0]
    y = r[1]
    fx = alfa * x - beta * x * y
    fy = gamma * x * y - delta * y
    return np.array([fx, fy], float)


a = 0  # punto inicial del intervalo
b = 30  # punto final del intervalo
r0 = np.array([2, 2], float)  # condiciones iniciales
N = 10000  # número de puntos

tp, xp, yp = runge_kutta_4_system_method(lokta_volterra, r0, a, b, N)
plt.plot(tp, xp, label="Población de conejos")
plt.plot(tp, yp, label='Población de zorros')
plt.title("Lokta-Volterra system")
plt.xlabel("t")
plt.ylabel("Miles de individuos")
plt.legend()
plt.show()
# -----------------------------------------Second order ODE-------------------------------------------------------------
"""
Ejercicio 7.6: el péndulo no lineal
Escribir un programa que permita describir el movimiento de un péndulo no lineal durante
4 periodos, con l = 10 cm que parte del reposo con un ángulo inicial de q = 179º usando el
método de Runge-Kutta de cuarto orden.

    d^2theta/dt^2 = -g/l * sin(theta) -> lo convertimos en un sistema
    
    dtheta/dt = omega
    domega/dt = -g/l * sin(theta)
"""

def pendulo(r):
    theta = r[0]
    omega = r[1]
    f_theta = omega
    f_omega = -g / l * np.sin(theta)
    return np.array([f_theta, f_omega], float)

a=0
b=10
N=1000
g,l = 9.81,0.1
r0=np.array([np.pi*179/180,0],float) # condiciones iniciales
tp, thetap ,omegap = runge_kutta_4_system_method(pendulo, r0, a, b, N)
xp = l*np.sin(thetap)
yp = -l*np.cos(thetap)

plt.plot(tp,thetap,"red")
plt.plot(tp,thetap,"b.")
plt.title("Pendulo no lineal")
plt.xlabel("t")
plt.ylabel("$\\theta$")
plt.show()

plt.plot(tp,omegap,"red")
plt.plot(tp,omegap,"b.")
plt.title("Pendulo no lineal")
plt.xlabel("t")
plt.ylabel("$\\omega$")
plt.show()

plt.plot(xp,yp)
plt.title("Pendulo no lineal")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#---------------------------------------------Adaptive methods----------------------------------------------------------
"""
Ejemplo 7.8: la órbita de un cometa
Resolver el problema con paso fijo y paso adaptado
    
    d^2x/dt^2 = -GMx/r^3
    d^2y/dt^2 = -GMy/r^3
"""

# definimos la función de nuestro sistema de ecuaciones diferenciales
def orbita_cometa(r):
    G=6.674e-11
    M=1.989e30
    x=r[0]
    y=r[1]
    vx=r[2]
    vy=r[3]
    r3=(x**2+y**2)**1.5
    fx=vx
    fy=vy
    fvx=-G*M*x/r3
    fvy=-G*M*y/r3
    return np.array([fx, fy, fvx, fvy], float)

# Metodo de Runge-Kutta de paso fijo
x0=4e12
y0=0
vx0=0
vy0=500
r0=np.array([x0,y0,vx0,vy0],float) # condiciones iniciales
tmax=3.2e9 # tiempo total simulacion
N = 1000

tp,xp,yp = runge_kutta_4_system_method(orbita_cometa, r0, 0, tmax, N)

plt.plot(xp,yp)
plt.title("Orbita de cometa paso fijo")
plt.show()

# Metodo de Runge-Kutta de paso adaptado
delta=1e3/365.25/24/3600
h0 = 1e5
xp,yp = runge_kutta_4_system_adaptive_method(orbita_cometa, r0, tmax, delta,h0)

plt.plot(xp,yp)
plt.title("Orbita de cometa paso adaptado")
plt.show()

#---------------------------------------------Other methods-------------------------------------------------------------
"""
Ejemplo 7.19: Método del salto de rana

    d^2x/dt^2 = (dx/dt)^2 - x - 5
    
usando el método del salto de la rana en el intervalo t e [0, 50], con un paso h=0.001 y usando las
condiciones iniciales x(0) = 1, v(0) = 0.
"""

def f_rana(r):
    x=r[0]
    v=r[1]
    fr=v
    fv=v**2-x-5
    return np.array([fr,fv],float)

# intervalo y paso
a=0
b=50
h=0.001
r1=np.array([1,0],float)

tp,xp = leapfrog_method(f_rana,r1,a,b,h)

plt.plot(tp,xp)
plt.title("Leapfrog method")
plt.xlabel("t")
plt.ylabel("x")
plt.show()


"""
Ejemplo 7.10: el método de Verlet y la órbita de la tierra
"""

from numpy.linalg import norm
G = 6.6738e-11
M = 1.9891e30
m = 5.9722e24

def orbita_tierra(r):
    return -G * M * r / norm(r) ** 3

def energia_potencial(r):
    return -G*M*m/norm(r)

def energia_cinetica(v):
    return 0.5*m*sum(v*v)

# Intervalo temporal
a = 0.0
b = 100e6
h = 3600.0

# Condiciones iniciales
x0 = 1.5710e11
y0 = 0.0
vx0 = 0.0
vy0 = 3.0278e4
r = np.array([x0,y0],float)
v = np.array([vx0,vy0],float)

tp,xp,yp, Vp,Tp ,Ep = verlet_method(orbita_tierra, energia_cinetica,energia_potencial,r,v,a,b,h)

plt.plot(xp,yp)
plt.title("órbita de tierra con Verlet")
plt.show()

plt.plot(tp,Vp, label = 'Energía potencial')
plt.plot(tp,Ep, label = 'Energía total')
plt.plot(tp,Tp, label = 'Energía cinética')
plt.title('Orbita tierra')
plt.legend()
plt.show()

plt.plot(tp,Ep,'g')
plt.show()

