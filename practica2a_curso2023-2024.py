"""
Práctica 2a: Obteniendo máximos y mínimos en dos dimensiones
con distintos métodos
"""

import numpy as np
# ----------------------------------------Ejercicio 1------------------------------------------------
# Definimos la función de dos variables
def function(x:float,y:float) -> float:
    return 10*np.sin(x)*np.sin(y) + x**2 + y**2

# Creamos un grid para la representación de la distribución de carga
N = 2000
xmax = 4
xmin = -4
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
xx, yy = np.meshgrid(x, y)
zz = function(xx,yy)

# Representación en 3D de la función
import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
surface = ax.plot_surface(xx,yy,zz, cmap='hsv')
fig.colorbar(surface,label="z")  # Agregar etiqueta a la barra de color
ax.set_title("Función de dos variables")  # Título del gráfico
ax.set_xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
ax.set_ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
ax.set_zlabel("z (eje del potencial)")     # Etiqueta para el eje z
plt.show()

# Representación en 2D de la función
plt.imshow(zz, origin='lower', cmap='hsv', extent=(-4, 4, -4, 4))
plt.colorbar(label="z")  # Agregar etiqueta a la barra de color
plt.title("Función de dos variables")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ----------------------------------------Ejercicio 2------------------------------------------------
# ----------------------------------------Ejercicio 3------------------------------------------------
# ----------------------------------------Ejercicio 4------------------------------------------------
# ----------------------------------------Ejercicio 5------------------------------------------------
# ----------------------------------------Ejercicio 6------------------------------------------------
# ----------------------------------------Ejercicio 7------------------------------------------------

