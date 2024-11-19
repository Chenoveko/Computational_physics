"""
Práctica 2a: Obteniendo máximos y mínimos en dos dimensiones
con distintos métodos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

from non_linear_equations import newton_raphson_multiple_variables, fixed_point_multiple_variables
from optimization import gauss_newton


# ----------------------------------------Ejercicio 1------------------------------------------------
# Definimos la función de dos variables
def f(r:np.array) -> np.array:
    return 10 * np.sin(r[0]) * np.sin(r[1]) + r[0] ** 2 + r[1] ** 2


# Creamos un grid para la representación de la funcion
N = 2000
xmax = 4
xmin = -4
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
xx, yy = np.meshgrid(x, y)
zz = f([xx, yy])

# Representación en 3D de la función
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
surface = ax.plot_surface(xx, yy, zz, cmap='hsv')
fig.colorbar(surface, label="z")  # Agregar etiqueta a la barra de color
ax.set_title("Función de dos variables")  # Título del gráfico
ax.set_xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
ax.set_ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
ax.set_zlabel("z (eje del potencial)")  # Etiqueta para el eje z
plt.show()

# Representación en 2D de la función
plt.imshow(zz, origin='lower', cmap='hsv', extent=(-4, 4, -4, 4))
plt.colorbar(label="z")  # Agregar etiqueta a la barra de color
plt.title("Función de dos variables")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ----------------------------------------Ejercicio 2------------------------------------------------
# Puntos iniciales para mínimos
x_inicial = np.array([2, -1, -2, 1.5],float)
y_inicial = np.array([2, 1, -2, -1.5],float)
x0 = np.array([[2,2],[-1,1],[-2,-2],[1.5,-1.5]],float)
# Representación en 2D de la función con mínimos y máximos iniciales
plt.imshow(zz, origin='lower', cmap='hsv', extent=(-4, 4, -4, 4))
plt.scatter(x_inicial,y_inicial, color='black')
plt.colorbar(label="z")  # Agregar etiqueta a la barra de color
plt.title("Función de dos variables con estimaciones de máximos y mínimos ")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()
# ----------------------------------------Ejercicio 3------------------------------------------------
"""
Newton-Raphson para encontrar extremos relativos --> Gauss-Newton
"""
def f_jacobiano(r):
    return [10*np.cos(r[0])*np.sin(r[1])+2*r[0],10*np.cos(r[1])*np.sin(r[1])-2*r[1]]

x0 = np.array([[2,2],[-1,1],[-2,-2],[1.5,-1.5]],float) # estimaciones iniciales

# Extremo usando la libreria scipy para comprobar resultados
extremo1=optimize.minimize(f,x0[0])
extremo2=optimize.minimize(f,x0[1])
extremo3=optimize.minimize(f,x0[2])
extremo4=optimize.minimize(f,x0[3])

print('----------Extremos usando optimize.minimize de scipy----------')
print('Extremo 1: ','x = ',extremo1.x[0],', y = ', extremo1.x[1],', z = ', extremo1.fun)
print('Extremo 2: ','x = ',extremo2.x[0],', y = ', extremo2.x[1],', z = ', extremo2.fun)
print('Extremo 3: ','x = ',extremo3.x[0],', y = ', extremo3.x[1],', z = ', extremo3.fun)
print('Extremo 4: ','x = ',extremo4.x[0],', y = ', extremo4.x[1],', z = ', extremo4.fun)


# ----------------------------------------Ejercicio 4------------------------------------------------
"""
Ejercicio 3 utilizando el método del pto fijo
    x = -5 * sin(y) * cos(x) = f_1(x,y)
    y = 5 * sin(x) * cos(y) = f_2(x,y)
    J = (f_1x , f_1y
         f_2x , f_2y)
"""
def f_1(r):
    return -5 * np.sin(r[1])*np.cos(r[0])

def f_1x(r):
    return 5*np.sin(r[1])*np.sin(r[0])

def f_1y(r):
    return -5*np.cos(r[1])*np.cos(r[0])

def f_2(r):
    return 5 * np.sin(r[0])*np.cos(r[1])

def f_2x(r):
    return 5 * np.cos(r[0])*np.cos(r[1])

def f_2y(r):
    return -5 * np.sin(r[0])*np.sin(r[1])

def sistema(r):
    return [f_1(r), f_2(r)]

def jacobiano_sistema(r):
    return [[f_1x(r),f_1y(r)],[f_2x(r),f_2y(r)]]

eps = 1e-6
x1,it = fixed_point_multiple_variables(sistema,x0,eps)
# ----------------------------------------Ejercicio 5------------------------------------------------
# ----------------------------------------Ejercicio 6------------------------------------------------
# ----------------------------------------Ejercicio 7------------------------------------------------
