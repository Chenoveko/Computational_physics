"""
Práctica 1: Obteniendo el potencial y el campo eléctrico a partir
de una distribución de carga
"""

import numpy as np
# ----------------------------------------Ejercicio 1------------------------------------------------
from basics import cartesianas_2_polares
import matplotlib.pyplot as plt


# Definimos la función de la densidad de carga
def densidad_carga(x: 'Posición x', y: 'Posición y',
                   rho_0: 'Factor de escala' = 100) -> 'Densidad de carga eléctrica':
    # Cambio coordenadas: cartesianas -> polares
    r, phi = cartesianas_2_polares(x, y)
    if 1 >= r >= 0:  # Por este if voy a tener que vectorizar la función para poder representarla en  el meshgrid
        return rho_0 * np.exp(-1 / (1 - r ** 3)) * np.sin(8 * phi)
    else:
        return 0


# Creamos un grid para la representación de la distribución de carga
N = 1000
xmax = 1
xmin = -1
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
xx, yy = np.meshgrid(x, y)

# Vectorizar la función para que acepte arrays
densidad_carga_vectorizada = np.vectorize(densidad_carga)

# Calcular la distribución de carga en la malla
mi_distribucion = densidad_carga_vectorizada(xx, yy)

# Dibujo la distribución de carga en el meshgrid con el patron de color seismic
plt.imshow(mi_distribucion, origin='lower', cmap='seismic', extent=(-1, 1, -1, 1))
plt.colorbar(label="Densidad de Carga")  # Agregar etiqueta a la barra de color
plt.title("Distribución de Carga")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ----------------------------------------Ejercicio 2------------------------------------------------
""" 
 - utilizaré la cuadratura gaussiana 
 - Integro solo donde hay carga -> r :[0,1]
"""

from gaussxw import gaussxw

# Definimos constantes y calculamos los zeros del polinomio de Legendre y los␣ pesos en el intervalo canónico
N_cuadraturas = 300  # Número de puntos de la cuadratura
ptos, w = gaussxw(N_cuadraturas)  # puntos y pesos de la cuadratura gaussiana canónica


# Definimos el potencial calculando la integral con la cuadratura gaussiana
def potencial_electrico(x: 'Posición x', y: 'Posición y') -> 'Potencial eléctrico':
    permitividad_vacio = 8.85e-12
    # Integral bidimensional rectangular
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            carga = densidad_carga(ptos[i], ptos[j])
            distancia = np.sqrt((x - ptos[i]) ** 2 + (y - ptos[j]) ** 2)
            I += w[i] * w[j] * carga / distancia
    return I / (4 * np.pi * permitividad_vacio)


# ----------------------------------------Ejercicio 3------------------------------------------------

"""
IMPORTANTE: No hace falta vectorizar el potencial, ya esta vectorizado porque solo estamos utilizando
operadores básicos (+,-,*,/,**...) y funciones de numpy. Si se vectoriza se vuelve muy ineficiente.
"""
# Creamos un grid para la representación del potencial
N = 200
xmax = 2
xmin = -2
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
xx, yy = np.meshgrid(x, y)

# Calcular la distribución de carga en la malla
mi_potencial = potencial_electrico(xx, yy)

# Dibujo el potencial eléctrico en el meshgrid con los límites de potencial
v_max, v_min = 10e11, -10e11
plt.imshow(mi_potencial, origin='lower', cmap='seismic', extent=(-2, 2, -2, 2), vmin=v_min, vmax=v_max)
plt.colorbar(label="Potencial eléctrico")  # Agregar etiqueta a la barra de color
plt.title("Potencial eléctrico")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ----------------------------------------Ejercicio 4------------------------------------------------

# Creamos un grid para la representación del campo eléctrico
N = 60
xmax = 2
xmin = -2
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
xx, yy = np.meshgrid(x, y)

# Calcular el campo eléctrico en  la malla
mi_potencial = potencial_electrico(xx, yy)
h = (xmax - xmin) / N
Ex = np.zeros([N, N])
Ey = np.zeros([N, N])
Ex[:, 0] = -(mi_potencial[:, 1] - mi_potencial[:, 0]) / h
Ex[:, N - 1] = -(mi_potencial[:, N - 1] - mi_potencial[:, N - 2]) / h
Ey[0, :] = -(mi_potencial[1, :] - mi_potencial[0, :]) / h
Ey[N - 1, :] = -(mi_potencial[:, N - 1] - mi_potencial[:, N - 2]) / h
Ex[:, 1:N - 1] = -(mi_potencial[:, 2:N] - mi_potencial[:, 0:N - 2]) / (2 * h)
Ey[1:N - 1, :] = -(mi_potencial[2:N, :] - mi_potencial[0:N - 2, :]) / (2 * h)

# ----------------------------------------Ejercicio 5------------------------------------------------
plt.quiver(xx, yy, Ex, Ey)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Campo Eléctrico derivado del Potencial")
plt.show()

# ----------------------------------------Ejercicio 6 (extra)------------------------------------------------
C = 10e11
Ex[Ex > C] = C
Ex[Ex < -C] = -C
Ey[Ey > C] = C
Ey[Ey < -C] = -C
plt.quiver(xx, yy, Ex, Ey)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Campo Eléctrico derivado del Potencial (evitando errores numéricos)")
plt.show()
