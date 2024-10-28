"""
Práctica 1: Obteniendo el potencial y el campo eléctrico a partir
de una distribución de carga
"""

import numpy as np
#----------------------------------------Ejercicio 1------------------------------------------------
from basics import cartesianas_2_polares
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)

xx, yy = np.meshgrid(x, y)

def distribucion_carga(x, y):
    rho_0 = 100
    # Cambio coordenadas: cartesianes -> polares
    r, phi = cartesianas_2_polares(x, y)
    if 1 > r >= 0:
        return rho_0 * np.exp(-1 / (1 - r ** 3)) * np.sin(8 * phi)
    else:
        return 0


# Vectorizar la función para que acepte arrays
distribucion_carga_vectorizada = np.vectorize(distribucion_carga)

# Calcular la distribución de carga en la malla
mi_distribucion = distribucion_carga_vectorizada(xx,yy)
print(mi_distribucion)

# Dibujo la distribución de carga en el meshgrid con el patron de color seismic
plt.imshow(mi_distribucion, cmap='seismic', extent=(-1, 1, -1, 1))
plt.colorbar(label="Densidad de Carga")  # Agregar etiqueta a la barra de color
plt.title("Distribución de Carga")       # Título del gráfico
plt.xlabel("x (eje de las abscisas)")    # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")   # Etiqueta para el eje y
plt.show()

#----------------------------------------Ejercicio 2------------------------------------------------
""" utilizaré el método de Romberg por esto:
- Función suave
- Puntos equiespaciados
"""

from integrals import romberg_rule

def potencial_electrico(x,y):
    permitividad_vacio = 8.85e-12



print(potencial_electrico(x,y))



#----------------------------------------Ejercicio 3------------------------------------------------
#----------------------------------------Ejercicio 4------------------------------------------------
#----------------------------------------Ejercicio 5------------------------------------------------
#----------------------------------------Ejercicio 6------------------------------------------------

