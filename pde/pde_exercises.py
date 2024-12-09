import numpy as np
from pde import *
import matplotlib.pyplot as plt
from data.banded import banded

# --------------------------------------------Equations with boundary conditions-----------------------------------------
""""
Ejemplo 8.1: Ecuación de Laplace en 2 dimensiones con el método de diferencias finitas
"""
# Constantes
M = 100  # número de cuadrados por lado
V = 1.0  # Voltaje en la parte de arriba de la caja
eps = 1e-6  # objetivo de precisión

phi = np.zeros([M + 1, M + 1], float)  # Estimación inicial.
phi[0, :] = V  # Fijamos los valores de la frontera

miphi, it = finite_differences_laplace_equation(phi, eps)

print(it)
plt.imshow(miphi)
plt.title('Potencial ejemplo 8.1')
plt.show()

"""
Ejemplo 8.2: Ecuación de Laplace en 2 dimensiones con el método de Gauss-Seidel sobrerrelajado
"""

# Constantes del problema
L = 1.0
M = 100
V = 1.0
eps = 1e-6

phi = np.zeros([M + 1, M + 1], float)  # Estimación inicial.
phi[0, :] = V  # Fijamos los valores de la frontera

miphi, it = gauss_seidel_overelaxed_laplace(phi, eps)

plt.imshow(miphi)
plt.title('Potencial ejemplo 8.2')
plt.show()

# --------------------------------------------Equations with initial conditions------------------------------------------

"""
Ejemplo 8.3: solución para la ecuación del calor con el método FTCS
"""

# Constantes
L = 0.01  # espesor de la barra en metros
D = 4.25e-6  # coeficiente de difusión térmica del acero
N = 100  # número de puntos en x, posición de la sección de espesor L.
a = L / N  # espaciado de la malla espacial
h = 1e-4  # paso en la variable temporal tiempo

k = h * D / a ** 2  # constante para la solución de Euler de la parte temporal

Tlo = 0.0  # temperatura baja en C
Tmid = 20.0  # temperatura inical del acero en C
Thi = 50.0  # Temperatura alta en C

# Array de tiempos en los que vamos a calcular el perfil de temperaturas
tp = [0.01, 0.1, 0.4, 1.0, 10.0]

# Inicializamos nuestra array de temperaturas
T = np.empty(N + 1, float)  # inicialización
# Fijamos condiciones de contorno
T[0] = Thi  # temperatura de la parte de la barra en contacto con el baño caliente
T[N] = Tlo  # temperatura de la parte de la barra en contacto con el baño frio
T[1:N] = Tmid  # temperatura inicial para el resto de la barra de acero

x = np.linspace(0, 1, 101)  # array con los puntos del grosor del acero

# Loop principal

eps = h / 1000  # precisión para definir un punto de la parte temporal
t = 0  # tiempo inicial
tf = 10 + eps

while t < tf:
    T[1:N] += k * (T[2:N + 1] + T[0:N - 1] - 2 * T[1:N])
    t += h
    # representamos el perfil de temperatura en los tiempos determinados
    for i in tp:
        if abs(t - i) < eps:
            plt.plot(x, T, label="T=%0.2f" % i)
plt.xlabel("x")
plt.ylabel("T")
plt.title("Perfil de temperaturas")
plt.legend()
plt.show()

"""
Ejemplo 8.4: solución de la ecuación de ondas con el método FTCS
"""

# Constantes
L = 1.0  # constantes de la cuerda
v = 100.0
d = 0.1
C = 1.0
sigma = 0.3
N = 100  # intevarlo espacial
a = L / N
h = 1e-6  # paso temporal


# función que nos describe la evolución espacial de la cuerda
def f(y):
    res = np.empty(N + 1, float)
    res[1:N] = (y[0:N - 1] + y[2:N + 1] - 2 * y[1:N]) * v * v / a / a
    res[0] = res[N] = 0.0
    return res


# Creamos el array inicial de $y$ y de $z$
x = np.linspace(0.0, L, N + 1)
psi = np.zeros(N + 1, float)
dpsi = C * x * (L - x) * np.exp(-(x - d) ** 2 / (2 * sigma * sigma)) / (L * L)
# pintamos las condiciones inciales
plt.plot(x, psi)
plt.plot(x, dpsi)
plt.show()

"""
Ejemplo 8.5: resolviendo la ecuación de Schrödinger con el método de Crank-Nicolson
"""

# Constantes
L = 2.0e-9  # longidut de la caja
N = 1000  # número de divisiones en x
a = L / N  # espaciado espacial
h = 1e-18  # paso temporal
m = 9.109e-31  # masa del electrón
hbar = 1.055e-34  # valor de hbar

# parametros para las condiciones iniciales
x0 = L / 2
sigma = 1.0e-10
kappa = 5.0e10
C = 1j * hbar / (4 * m * a * a)
a1 = 1 + 2 * h * C
a2 = -h * C
b1 = 1 - 2 * h * C
b2 = h * C

# 1. usando el programa banded
# Creamos el array tridiagonal A, vamos a usar el programa banded del tema 5
A = np.empty([3, N - 1], complex)
A[0, :] = a2  # primera fila con a2
A[1, :] = a1  # segunda fila con a1
A[2, :] = a2  # tercera fila con a2
# Inicializamos el array para x y psi
x = np.linspace(0, L, N + 1)
psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * kappa * x)
psi[0] = psi[N] = 0
# Bucle principal del metodo Crank-Nicolson
# calculamos el producto de B \psi, como B es tridiagonal es muy fácil
v = b2 * psi[0:N - 1] + b1 * psi[1:N] + b2 * psi[2:N + 1]
psi[1:N] = banded(A, v, 1, 1)  # nuestra solución viene de resolver el problema lineal
plt.plot(x, np.real(psi))
plt.show()


# definimos una función que nos evoluciona la función de ondas hasta un tiempo dado
def psi_evol(tmax):
    t = 0
    # Inicializamos el array para x y psi
    x = np.linspace(0, L, N + 1)
    psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * kappa * x)
    psi[0] = psi[N] = 0
    # Bucle principal del metodo Crank-Nicolson
    while t < tmax:
        # calculamos el producto de B \psi, como B es tridiagonal es muy fácil
        v = b2 * psi[0:N - 1] + b1 * psi[1:N] + b2 * psi[2:N + 1]
        psi[1:N] = banded(A, v, 1, 1)
        t += h
    return psi


plt.plot(x, np.real(psi_evol(1 * h)))
plt.show()

for i in range(100, 500, 100):
    plt.plot(x, np.real(psi_evol(i * h)))
plt.show()
