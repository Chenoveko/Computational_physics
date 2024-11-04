"""
Práctica 1: Obteniendo el potencial y el campo eléctrico a partir
de una distribución de carga
"""

import numpy as np
# ----------------------------------------Ejercicio 1------------------------------------------------
from basics import cartesianas_2_polares
import matplotlib.pyplot as plt


# Definimos la función de la densidad de carga
def densidad_carga(r, phi):
    rho_0 = 100
    if r <= 1:  # Por este if voy a tener que vectorizar la función para poder representarla en  el meshgrid
        return rho_0 * np.exp(-1 / (1 - r ** 3)) * np.sin(8 * phi)
    else:
        return 0


# Creamos un grid para la representación de la distribución de carga
N = 1000
xmax = 1
xmin = -1
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
X, Y = np.meshgrid(x, y)
R, PHY = cartesianas_2_polares(X, Y)

# Vectorizar la función para que acepte arrays
densidad_carga_vectorizada = np.vectorize(densidad_carga)

# Calcular la distribución de carga en la malla
mi_distribucion = densidad_carga_vectorizada(R, PHY)

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

from gaussxw import gaussxwab


# Definimos el potencial calculando la integral con la cuadratura gaussiana
def potencial_electrico(r):
    permitividad_vacio = 8.85e-12
    # Definimos constantes y calculamos los zeros del polinomio de Legendre y los pesos
    N_cuadraturas = 50  # Número de puntos de la cuadratura
    r_prima, w_r = gaussxwab(N_cuadraturas, 0, 1)
    phy_prima, w_phy = gaussxwab(N_cuadraturas, 0, 2 * np.pi)

    def integrando(r, r_prima, phy_prima):
        return densidad_carga_vectorizada(r_prima, phy_prima) * r / abs(r - r_prima)

    # Integral bidimensional
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            I += w_r[i] * w_phy[j] * integrando(r, r_prima[i], phy_prima[j])
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
X, Y = np.meshgrid(x, y)
R, PHY = cartesianas_2_polares(X, Y)
mi_potencial = potencial_electrico(R)

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
X, Y = np.meshgrid(x, y)
R, PHY = cartesianas_2_polares(X, Y)

# Calcular el campo eléctrico en la malla
from derivatives import centered_derivative

E_r = centered_derivative(potencial_electrico, R)
E_x = E_r * np.cos(PHY)
E_y = E_r * np.sin(PHY)

# ----------------------------------------Ejercicio 5------------------------------------------------
plt.quiver(X, Y, E_x, E_y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Campo Eléctrico derivado del Potencial")
plt.show()

# ----------------------------------------Ejercicio 6 (extra)------------------------------------------------
C = 10e11
E_x[E_x > C] = C
E_x[E_x < -C] = -C
E_y[E_y > C] = C
E_y[E_y < -C] = -C
plt.quiver(X, Y, E_x, E_y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Campo Eléctrico derivado del Potencial (evitando errores numéricos)")
plt.show()
