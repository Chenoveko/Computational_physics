"""
Práctica 2a: Obteniendo máximos y mínimos en dos dimensiones
con distintos métodos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# ----------------------------------------Ejercicio 1-------------------------------------------------------------------
"""
Representar la función para −4 ≤ 𝑥 ≤ 4, −4 ≤ 𝑦 ≤ 4 en 3 dimensiones, y también en
2 dimensiones con imshow, en un grid 2000 × 2000.
"""


# Definimos la función de dos variables
def f(x, y):
    return 10 * np.sin(x) * np.sin(y) + x ** 2 - y ** 2


# Definimos la función de dos variables con arrays
def f_array(x):
    return 10 * np.sin(x[0]) * np.sin(x[1]) + x[0] ** 2 - x[1] ** 2


# Creamos un grid para la representación de la funcion
N = 2000
xmax = 4
xmin = -4
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
xx, yy = np.meshgrid(x, y)
zz = f(xx, yy)

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

# ----------------------------------------Ejercicio 2-------------------------------------------------------------------
"""
Elegir puntos iniciales para calcular los máximos y mínimos relativos de la función en
la región −4 ≤ 𝑥 ≤ 4, −4 ≤ 𝑦 ≤ 4, y añadirlos a la representación de densidad.
Ayuda: Se recomienda utilizar en imshow la opción extent para representar con facilidad
los puntos iniciales escogidos.
"""
# Puntos iniciales
x_inicial = np.array([2, -1, -2, 1], float)
y_inicial = np.array([1, 2, -1, -2], float)

# Representación en 2D de la función con mínimos y máximos iniciales
plt.imshow(zz, origin='lower', cmap='hsv', extent=(-4, 4, -4, 4))
plt.scatter(x_inicial, y_inicial, color='black')
plt.colorbar(label="z")  # Agregar etiqueta a la barra de color
plt.title("Función de dos variables con estimaciones de máximos y mínimos ")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ----------------------------------------Ejercicio 3-------------------------------------------------------------------
"""
Partiendo de los puntos iniciales anteriores, calcular los extremos relativos de la
función con el método de Newton-Rapson (hacer uso de alguno de los métodos de resolución
de ecuaciones lineales estudiados en clase) aplicado sobre las condiciones de extremo relativo.
Añadir a la representación de densidad los extremos obtenidos. ¿Cuántas iteraciones son
necesarias para alcanzar cada extremo?

Newton-Raphson para encontrar extremos relativos --> Gauss-Newton

Las ecuaciones a resolver son
    df/dx -> 10 cos 𝑥 sin 𝑦 + 2𝑥 = 0
    df/dy -> 10 sin 𝑥 cos 𝑦 − 2𝑦 = 0
"""


# Definimos nuestro sistema de ecuaciones
def sistema(x):
    return [10 * np.cos(x[0]) * np.sin(x[1]) + 2 * x[0], 10 * np.sin(x[0]) * np.cos(x[1]) - 2 * x[1]]


# Definimos el jacobiano del sistema
def jacobiano_sistema(x):
    return [[-10 * np.sin(x[0]) * np.sin(x[1]) + 2, 10 * np.cos(x[0]) * np.cos(x[1])],
            [10 * np.cos(x[0]) * np.cos(x[1]), -10 * np.sin(x[0]) * np.sin(x[1]) - 2]]


def gauss_newton(sistema, jacobiano, x0, prec):
    x1 = np.copy(x0)
    it = np.zeros(len(x0))
    pos = 0
    # Para cada estimación inicial aplicamos Newton-Raphson
    for i in x1:
        err = 1
        while err > prec:
            dx = np.linalg.solve(jacobiano(i), sistema(i))
            i -= dx
            err = abs(max(dx))
            it[pos] += 1
        x1[pos] = i
        pos += 1
    plt.plot(x0[:, 0], x0[:, 1], 'ko')
    plt.plot(x1[:, 0], x1[:, 1], 'r*')

    for k in range(len(x1)):
        print('Se han necesitado', str(int(it[k])), 'iteraciones para llegar al siguiente extremo relativo: ')
        print(x1[k])
        print()
    return x1, it


# estimaciones iniciales y precisión
x0 = np.array([[2, 1], [-1, 2], [-2, -1], [1, -2]], float)
prec = 1e-6
plt.imshow(zz, origin='lower', cmap='hsv', extent=(-4, 4, -4, 4))
extremos, it = gauss_newton(sistema, jacobiano_sistema, x0, prec)
plt.colorbar(label="z")  # Agregar etiqueta a la barra de color
plt.title("Con Gauss-Newton")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

print(extremos)
print('----------Extremos usando Gauss-Newton-------------')
print('Extremo 1: ', 'x = ', extremos[0][0], ', y = ', extremos[0][1], ', z = ', f_array(extremos[0]))
print('Extremo 2: ', 'x = ', extremos[1][0], ', y = ', extremos[1][1], ', z = ', f_array(extremos[1]))
print('Extremo 3: ', 'x = ', extremos[2][0], ', y = ', extremos[2][1], ', z = ', f_array(extremos[2]))
print('Extremo 4: ', 'x = ', extremos[3][0], ', y = ', extremos[3][1], ', z = ', f_array(extremos[3]))


# Extremo usando la libreria scipy para comprobar resultados
# Definimos la función de dos variables con arrays
def f_array_minimize(x):
    return 10 * np.sin(x[0]) * np.sin(x[1]) + x[0] ** 2 - x[1] ** 2


def f_array_maximize(x):
    return -1 * (10 * np.sin(x[0]) * np.sin(x[1]) + x[0] ** 2 - x[1] ** 2)


extremo1 = optimize.minimize(f_array_maximize, x0[0])  # máximo
extremo2 = optimize.minimize(f_array_minimize, x0[1])  # mínimo
extremo3 = optimize.minimize(f_array_maximize, x0[2])  # máximo
extremo4 = optimize.minimize(f_array_minimize, x0[3])  # mínimo

print('----------Extremos usando optimize.minimize de scipy----------')
print('Extremo 1: ', 'x = ', extremo1.x[0], ', y = ', extremo1.x[1], ', z = ', -1 * extremo1.fun)
print('Extremo 2: ', 'x = ', extremo2.x[0], ', y = ', extremo2.x[1], ', z = ', extremo2.fun)
print('Extremo 3: ', 'x = ', extremo3.x[0], ', y = ', extremo3.x[1], ', z = ', -1 * extremo3.fun)
print('Extremo 4: ', 'x = ', extremo4.x[0], ', y = ', extremo4.x[1], ', z = ', extremo4.fun)

# ----------------------------------------Ejercicio 4-------------------------------------------------------------------
"""
Partiendo de los mismos puntos iniciales, tratar de calcular los máximos y mínimos
de la función con el método del punto fijo. Añadir a la representación de densidad los extremos
obtenidos. ¿Converge el método?
Ayuda: Se recomienda calcular un número fijo de iteraciones como máximo.

Las ecuaciones se pueden reescribir como:
    𝑥 = −5 cos 𝑥 sin 𝑦
    𝑦 = 5 sin 𝑥 cos 𝑦
"""
from non_linear_equations import pto_fijo_2_variables

def sistema_pto_fijo(x):
    return [-5 * np.cos(x[0]) * np.sin(x[1]), 5 * np.sin(x[0]) * np.cos(x[1])]

prec = 1e-3

plt.imshow(zz, origin='lower', cmap='hsv', extent=(-4, 4, -4, 4))
x2 = pto_fijo_2_variables(sistema, x0, prec)

plt.plot(x0[:, 0], x0[:, 1], 'ko')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.title('Con punto fijo')
plt.colorbar()
plt.show()
print('Los extremos relativos con el pto fijo no convergen')
print(x2)
# ----------------------------------------Ejercicio 5-------------------------------------------------------------------
"""
Demostrar por qué converge o no el método del punto fijo utilizado en el
apartado anterior. ¿Y si reescribes de otra forma?

Para probar si converge o no el método en un punto, hay que calcular la matriz jacobiana asociado a las ecuaciones de punto
fijo, y calcular el mayor de sus autovalores en valor absoluto. El método converge si y solo si dicha cantidad es menor que 1.
"""


def jacobiano_pto_fijo(x):
    return [[5 * np.sin(x[0]) * np.sin(x[1]), -5 * np.cos(x[0]) * np.cos(x[1])],
            [5 * np.cos(x[0]) * np.cos(x[1]), -5 * np.sin(x[0]) * np.sin(x[1])]]


# Podemos evaluarla en las soluciones que nos dio Newton Raphson
print(np.linalg.eigvals(jacobiano_pto_fijo(extremos[0])))
print(np.linalg.eigvals(jacobiano_pto_fijo(extremos[1])))
print(np.linalg.eigvals(jacobiano_pto_fijo(extremos[2])))
print(np.linalg.eigvals(jacobiano_pto_fijo(extremos[3])))
# Vemos que el metodo no converge para ninguno de los puntos
