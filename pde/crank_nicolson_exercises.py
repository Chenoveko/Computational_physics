import numpy as np
from data.banded import banded
from matplotlib import pyplot as plt

""" 
Ejemplo 8.5: resolviendo la ecuación de Schrödinger con el método de Crank-Nicolson
"""

# Constantes
L = 2.0e-9  # longitud de la caja
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

# Creamos el array tridiagonal A, vamos a usar el programa banded del tema 5
A = np.empty([3, N - 1], complex)
A[0, :] = a2  # primera fila con a2
A[1, :] = a1  # segunda fila con a1
A[2, :] = a2  # tercera fila con a2

# Inicializamos el array para x y psi
x = np.linspace(0, L, N + 1)
psi = np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) * np.exp(1j * kappa * x) # condición inicial
psi[0] = psi[N] = 0 # condición de frontera

# Bucle principal del metodo Crank-Nicolson

# calculamos el producto de B \psi, como B es tridiagonal es muy fácil
v = b2 * psi[0:N - 1] + b1 * psi[1:N] + b2 * psi[2:N + 1]
psi[1:N] = banded(A, v, 1, 1)  # nuestra solución viene de resolver el problema lineal

plt.plot(x, np.real(psi))
plt.title('Función de onda inicial')
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
plt.title('Evolución instante h después')
plt.show()

for i in range(100, 500, 100):
    plt.plot(x, np.real(psi_evol(i * h)))
plt.title('Evolución')
plt.show()