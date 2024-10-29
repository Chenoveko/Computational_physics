"""
Práctica 1: Obteniendo el potencial y el campo eléctrico a partir
de una distribución de carga
"""

import numpy as np
# ----------------------------------------Ejercicio 1------------------------------------------------
from basics import cartesianas_2_polares
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)

xx, yy = np.meshgrid(x, y)

def distribucion_carga(x, y):
    rho_0 = 100
    # Cambio coordenadas: cartesianas -> polares
    r, phi = cartesianas_2_polares(x, y)
    if 1 > r >= 0:
        return rho_0 * np.exp(-1 / (1 - r ** 3)) * np.sin(8 * phi)
    else:
        return 0

# Vectorizar la función para que acepte arrays
distribucion_carga_vectorizada = np.vectorize(distribucion_carga)

# Calcular la distribución de carga en la malla
mi_distribucion = distribucion_carga_vectorizada(xx, yy)

# Dibujo la distribución de carga en el meshgrid con el patron de color seismic
plt.imshow(mi_distribucion, cmap='seismic', extent=(-1, 1, -1, 1))
plt.colorbar(label="Densidad de Carga")  # Agregar etiqueta a la barra de color
plt.title("Distribución de Carga")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ----------------------------------------Ejercicio 2------------------------------------------------
""" 
 - utilizaré la cuadratura gaussiana 
 - Integro solo donde hay carga -> x:[-1,1], y:[-1,1]
"""

from gaussxw import gaussxwab

N_cuadraturas = 10  # Número de puntos de la cuadratura
ptos, w = gaussxwab(N_cuadraturas, -1, 1)  # puntos y pesos de la cuadratura gaussiana

def potencial_electrico(x: 'Posición x', y: 'Posición y') -> 'Potencial eléctrico':
    permitividad_vacio = 8.85e-12
    # Integral bidimensional rectangular
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            I += w[i] * w[j] * distribucion_carga(ptos[i], ptos[j]) / np.sqrt(
                (x - ptos[i]) ** 2 + (y - ptos[i] - y) ** 2)
    return I * 1 / (4 * np.pi * permitividad_vacio)

# ----------------------------------------Ejercicio 3------------------------------------------------
N = 200
x = np.linspace(-2, 2, N)
y = np.linspace(-2, 2, N)
xx, yy = np.meshgrid(x, y)

# Vectorizar la función para que acepte arrays
potencial_electrico_vectorizado = np.vectorize(potencial_electrico)

# Calcular la distribución de carga en la malla
mi_potencial = potencial_electrico_vectorizado(xx, yy)
print(mi_potencial)

# Dibujo el potencial eléctrico en el meshgrid con los límites de potencial
v_max, v_min = 10e-11, -10e11
plt.imshow(mi_potencial, extent=(-2, 2, -2, 2), vmax=v_max, vmin=v_min)
plt.colorbar(label="Potencial eléctrico")  # Agregar etiqueta a la barra de color
plt.title("Potencial eléctrico")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ----------------------------------------Ejercicio 4------------------------------------------------
# ----------------------------------------Ejercicio 5------------------------------------------------
# ----------------------------------------Ejercicio 6------------------------------------------------
