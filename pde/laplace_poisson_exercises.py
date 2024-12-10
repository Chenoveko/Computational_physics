import numpy as np
import matplotlib.pyplot as plt
from data.banded import banded
from pde import *

""""
Ejemplo 8.1: Ecuación de Laplace en 2 dimensiones con el método de diferencias finitas (Jacobi)
"""
# Constantes
M = 100  # número de cuadrados por lado
V = 1.0  # Voltaje en la parte de arriba de la caja
eps = 1e-6  # objetivo de precisión

phi = np.zeros([M + 1, M + 1], float)  # Estimación inicial.
phi[0, :] = V  # Fijamos los valores de la frontera en la estimación


laplace,it = jacobi_laplace(phi, eps)

plt.imshow(laplace)
plt.title('Método diferencias finitas / Jacobi')
plt.show()

"""
Ejemplo: ecuación de laplace con fronteras fijadas a V
"""
# Constantes
M = 100  # número de cuadrados por lado
V = 1.0  # Voltaje en la parte de arriba de la caja
eps = 1e-6  # objetivo de precisión

phi = np.zeros([M + 1, M + 1], float)  # Estimación inicial.
# Fijamos el valor V en la frontera
phi[:, 0] = V  # Fijo lado izquierdo
phi[:, M] = V  # Fijo lado derecho
phi[0, :] = V  # Fijo techo
phi[M, :] = V  # Fijo suelo

miphi,it = jacobi_laplace(phi, eps)

plt.imshow(miphi)
plt.title('Mi ejemplo')
plt.show()

"""
Ejemplo 8.2: Ecuación de Laplace en 2 dimensiones con el método sobrerelajación (Gauss-Seidel)
"""
# Constantes del problema
L = 1.0
M = 100
V = 1.0
eps = 1e-6
w = 0.94 # factor relajación

phi = np.zeros([M + 1, M + 1], float)  # Estimación inicial.
phi[0, :] = V  # Fijamos los valores de la frontera en la estimación

# bucle principal
delta = 1.0
it = 0
while delta>eps:
    delta=0
    #Calculamos los nuevos valores del potencial
    for i in range(1,M):
        for j in range(1,M):
            phip=phi[i,j]
            phi[i,j]=(1+w)*(phi[i+1,j]+phi[i-1,j]+phi[i,j+1]+phi[i,j-1])/ 4-w*phi[i,j]
            diff=abs(phi[i,j]-phip)
            delta=max([delta,diff])
    # Calculamos el maximo del cambio
    it+=1

plt.imshow(phi)
plt.title('Metodo sobrerelajación / Gauss-Seidel')
plt.show()

"""
Ejercicio 8.1: solución de la ecuación Poisson
"""
# Definimos constantes
L = 1.0  # longitud de la caja en metros
M = 100  # número de cubos en cada lado

rho0 = 1
eps = 5e-6  # precisión pedida

phi = np.zeros([M + 1, M + 1], float)  # Estimación inicial

# Con slicing definimos rho
rho = np.zeros([M + 1, M + 1], float)
rho[20:40,60:80] = rho0
rho[60:80,20:40] = -1*rho0

poisson, it = jacobi_poisson(rho,L,phi, eps)

plt.imshow(poisson)
plt.title('Ecuación de Poisson')
plt.show()