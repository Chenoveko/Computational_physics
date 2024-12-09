import numpy as np
import matplotlib.pyplot as plt
from integrals import *

# --------------------------------------Integrales definidas en una variable---------------------------------------------
"""
Ejemplo 4.2: aplicando la regla del trapecio
"""


def g(x):
    return x ** 4 - 2 * x + 1


print('Regla del trapecio ejemplo 4.2 = ', trapezium_rule(g, 0, 2, 10))

"""
Ejercicio 4.1: integrando valores discretos
"""

data_velocities = np.loadtxt('data/velocities.txt')
tiempos = data_velocities[:, 0]
velocidades = data_velocities[:, 1]

print('Distancia total recorrida con trapecio = ', trapezium_rule_discrete(velocidades, 0, 100), 'm')

plt.plot(tiempos, velocidades)
plt.title('Velocidad partícula')
plt.xlabel('tiempo [s]')
plt.ylabel('velocidad [m/s]')
plt.show()

distancia_acumulada = [0]

for i in range(1, 101):
    distancia_acumulada.append(trapezium_rule_discrete(velocidades[:i + 1], tiempos[0], tiempos[i]))

plt.plot(tiempos, distancia_acumulada)
plt.title('Distancia en función del tiempo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Distancia [m]')
plt.grid(True)
plt.show()
"""
Ejemplo 4.4: aplicando la regla de Simpson
"""

print('Regla de Simpson ejemplo 4.4 = ', simpson_rule(g, 0, 2, 10))

"""
Ejemplo 4.5: el límite difractivo de un telescopio
"""


# Escribe una función que calcule el valor de la función de Bessel usando la regla de Simpson con N=1000.
def bessel(x, m=1):
    def integrando(theta):
        return np.cos(m * theta - x * np.sin(theta))

    return 1 / np.pi * simpson_rule(integrando, 0, np.pi, 1000)


# Representa en una misma gráfica las funciones de Bessel J0, J1, J2 y J3 como función de x, en el intervalo [0, 50].
xp = np.linspace(0, 50, 1000)
j0 = bessel(xp, 0)
j1 = bessel(xp)
j2 = bessel(xp, 2)
j3 = bessel(xp, 3)

plt.plot(xp, j0, label='J_0')
plt.plot(xp, j1, label='J_1')
plt.plot(xp, j2, label='J_2')
plt.plot(xp, j3, label='J_3')
plt.legend()
plt.show()

"""
Escribir un segundo programa que genere una gráfica de densidad en un grid 100x100 de la intensidad de un patrón 
circular de una fuente luminosa con λ = 500 nm en un grid que cubra valores de r hasta 1µm
"""


def intensidad(r, lamb=500):
    k = 2 * np.pi / lamb
    return (1 / (k * r) * bessel(k * r)) ** 2


xp = np.linspace(-1000, 1000, 300)
yp = np.linspace(-1000, 1000, 300)
xx, yy = np.meshgrid(xp, yp)

rr = np.sqrt(xx ** 2 + yy ** 2)

plt.imshow(intensidad(rr), cmap='hot', vmin=0, vmax=0.001)
plt.show()

"""
Ejemplo 4.6: integrando con paso adaptado
"""


def integral(x):
    return np.sin(np.sqrt(100 * x)) ** 2


print('Trapecio paso adaptado', trapezium_rule_adaptive(integral, 0, 1, 1.0e-6))
print('Simpson paso adaptado', simpson_rule_adaptive(integral, 0, 1, 1.0e-6))
print('Romberg', romberg_rule(integral, 0, 1, 1.0e-6))

"""
Ejemplo 4.7: usando la cuadratura gaussiana
"""

print('Cuadratura gaussiana = ', gaussian_quadrature(g, 0, 2, 3))

# ----------------------------------Integrales sobre rangos infinitos----------------------------------------------------
"""
Ejemplo 4.9: integrando la gaussiana

    Tenemos que hacer un cambio de variable
"""


def gauss_integrand(x):
    return np.exp(-x ** 2)


def f(z):
    x = z / (1 - z ** 2)
    dxdz = (1 + z ** 2) / (1 - z ** 2) ** 2
    return dxdz * gauss_integrand(x)


print('Integrando la gaussiana = ', gaussian_quadrature(f, -1, 1, 50))

# ------------------------------------------Integrales múltiples---------------------------------------------------------
"""
Ejemplo 4.10: Atracción gravitatoria de una lámina metálica

    Integrales bidimensionales rectangulares
"""

# Constantes del problema
L = 10  # lado de la placa
G = 6.674e-11  # constante de la gravitación universal
N = 100  # número de puntos de la cuadratura
rho = 100  # densidad por unidad de superficie
# Definimos los valores de z
zmin = 0
zmax = 1
# Puntos y pesos de la cuadratura gaussiana
x, w = gaussxwab(N, -L / 2, L / 2)


# hacemos la integral
def Fz_lamina(z):
    I = 0
    for i in range(N):
        for j in range(N):
            I += w[i] * w[j] / (x[i] ** 2 + x[j] ** 2 + z ** 2) ** (3 / 2)
    return G * rho * z * I


# Representamos la fuerza para cada valor de z
zpoints = np.linspace(zmin, zmax, 100)
fzpoints = list(map(Fz_lamina, zpoints))
plt.plot(zpoints, fzpoints)
plt.show()
