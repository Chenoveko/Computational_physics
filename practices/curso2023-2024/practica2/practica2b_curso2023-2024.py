"""
Práctica 2a: Obteniendo máximos y mínimos en dos dimensiones
con distintos métodos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from optimization import razon_aurea_maximizar

# --------------------------------------------------Ejercicio 1----------------------------------------------------------
"""
Importar y representar el fichero de datos en 3 dimensiones, y también en 2 dimensiones
con imshow.
"""
# Creamos un grid para la representación de los datos
N = 121
xmax = 3
xmin = -3
x = np.linspace(xmin, xmax, N)
y = np.linspace(xmin, xmax, N)
data_x, data_y = np.meshgrid(x, y)
data_z = np.loadtxt('data_practica_2.txt', float)  # importamos los datos

# Representación en 3D de la función
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
surface = ax.plot_surface(data_x, data_y, data_z, cmap='terrain')
fig.colorbar(surface, label="z")  # Agregar etiqueta a la barra de color
ax.set_title("Altura montañas")  # Título del gráfico
ax.set_xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
ax.set_ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
ax.set_zlabel("z (altura)")  # Etiqueta para el eje z
plt.show()

# Representación en 2D de la función
plt.imshow(data_z, origin='lower', cmap='terrain', extent=(-3, 3, -3, 3))
plt.colorbar(label="z")  # Agregar etiqueta a la barra de color
plt.title("Altura montañas")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# --------------------------------------------------Ejercicio 2----------------------------------------------------------
""""
Definir una función interpoladora de los puntos del fichero por medio de splines
cúbicos. Para ello, hacer uso de la función RectBivariateSpline de scipy.interpolate
"""
f_z = RectBivariateSpline(x, y, data_z.T)
# --------------------------------------------------Ejercicio 3----------------------------------------------------------
"""
Realizar la representación de densidad de la función del apartado anterior en −3 ≤
𝑥 ≤ 3, −3 ≤ 𝑦 ≤ 3 en un grid 1000 × 1000. Además, elegir un punto inicial para calcular el
máximo global en dicho dominio, y añadirlo a la representación.
"""
N = 1000
xmax = 3
xmin = -3
x_prima = np.linspace(xmin, xmax, N)
y_prima = np.linspace(xmin, xmax, N)
pto_inicial_maximo_global = np.array([1.5, -1.4])
"""
Ayuda:
- En el imshow, basta introducir f(x’,y’).T donde x’ e y’ son los arrays creados para la representación (no hay que usar meshgrid).
- Se recomienda utilizar en imshow la opción extent para representar con facilidad el punto inicial escogido.
"""
# Representación en 2D de la función con punto inicial
plt.imshow(f_z(x_prima, y_prima).T, origin='lower', cmap='terrain', extent=(-3, 3, -3, 3))
plt.scatter(pto_inicial_maximo_global[0], pto_inicial_maximo_global[1])
plt.colorbar(label="z")  # Agregar etiqueta a la barra de color
plt.title("Altura montañas (con punto inicial)")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# --------------------------------------------------Ejercicio 4----------------------------------------------------------
"""
Implementar el método de máxima pendiente para funciones de dos variables con el
fin de encontrar un máximo con una precisión de 10−5. El cálculo del gradiente deberá hacerse
con diferencias centradas, con paso ℎ = 10−3 para ambas variables, y utilizar el método de
la razón áurea para la maximización en una variable. ¿Por qué es preferible al método de
Gauss-Newton? -> es preferible porque no se puede calcular la segunda derivada
"""
eps = 1e-5


def gradiente_diferencias_centradas(f, x, y):
    h = 1e-3
    grad = [0, 0]
    grad[0] = (f(x + h / 2, y) - f(x - h / 2, y)) / h
    grad[1] = (f(x, y + h / 2) - f(x, y - h / 2)) / h
    return grad


def maxima_pendiente(f: 'funcion', x0: 'pto x inicial', y0: 'pto y inicial', prec: 'precision objetivo') -> 'Extremo':
    err = 1
    cont = 0
    while err > prec:
        v = gradiente_diferencias_centradas(f, x0, y0)
        norm_v = (v[0] ** 2 + v[1] ** 2) ** (1 / 2)

        def f_2(k):
            return f(x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v)

        k = razon_aurea_maximizar(f_2, -1.5, 1.5,
                                  prec)  # Los valores de x1 y x4 hay que "tunearlos" dependiendo de la estimación inicial.
        x1, y1 = x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v
        err = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)
        cont += 1
        plt.plot(x1, y1, 'r.')
        plt.arrow(x0, y0, x1 - x0, y1 - y0, length_includes_head=True, head_width=0.1, head_length=0.05, color='k')
        x0, y0 = x1, y1
        print('Se han necesitado', str(cont), 'iteraciones para llegar al máximo:')
        print('x=', x1, ' , y=', y1, sep="")
        print()
    return x1, y1


# --------------------------------------------------Ejercicio 5----------------------------------------------------------
"""
Aplicar el método de la máxima pendiente a la función definida en el apartado 2
para obtener su máximo global en −3 ≤ 𝑥 ≤ 3, −3 ≤ 𝑦 ≤ 3. Añadir a la representación
de densidad del apartado 3 las iteraciones que se han obtenido (para ello, puede modificarse
levemente la función del apartado anterior).

Ayuda: Por el funcionamiento de RectBivariateSpline, al evaluar la función interpoladora
𝑓 en un punto, se nos proporciona la evaluación de la función dentro de un array. Por ello
se recomienda definir una función que evaluada en un punto (𝑥, 𝑦), devuelva 𝑓(𝑥, 𝑦)[0][0], y
maximizar dicha función.
"""
# Estimación inicial
x0, y0 = 1.5, -0.5


# Siguiendo la ayuda, vamos a maximizar la siguiente función
def f_z_3(x, y):
    return f_z(x, y)[0][0]


# Repetimos el plot del apartado 3
plt.imshow(f_z(x_prima, y_prima).T, cmap='terrain', extent=(-3, 3, -3, 3), origin='lower')
plt.plot(x0, y0, 'b.')
plt.colorbar()

xf, yf = maxima_pendiente(f_z_3, x0, y0, eps)
plt.plot(xf, yf, 'g.')  # Añadimos en verde el máximo obtenido
plt.title('Maxima pendiente')
plt.show()
# --------------------------------------------------Ejercicio 6---------------------------------------------------------
"""
Repetir el apartado anterior con otros puntos iniciales para obtener los otros máximos
relativos. ¿Qué pasa si se coge como punto inicial el origen?¿Y si nos desplazamos de él un
poco?
"""
# Repetimos el apartado anterior con puntos en torno al origen
# para alcanzar el resto de máximos relativos
x0, y0 = 0, 0
x01, y01 = 0.1, 0
x02, y02 = -0.1, 0
x03, y03 = 0, 0.1
x04, y04 = 0, -0.1
x05, y05 = -0.1, 0.1
plt.imshow(f_z(x_prima, y_prima).T, cmap='terrain', extent=(-3, 3, -3, 3), origin='lower')
plt.plot(x0, y0, 'b.')
plt.plot(x01, y01, 'b.')
plt.plot(x02, y02, 'b.')
plt.plot(x03, y03, 'b.')
plt.plot(x04, y04, 'b.')
plt.plot(x05, y05, 'b.')
plt.colorbar()
xf, yf = maxima_pendiente(f_z_3, x0, y0, eps)
plt.plot(xf, yf, 'g.')
xf, yf = maxima_pendiente(f_z_3, x01, y01, eps)
plt.plot(xf, yf, 'g.')
xf, yf = maxima_pendiente(f_z_3, x02, y02, eps)
plt.plot(xf, yf, 'g.')
xf, yf = maxima_pendiente(f_z_3, x03, y03, eps)
plt.plot(xf, yf, 'g.')
xf, yf = maxima_pendiente(f_z_3, x04, y04, eps)
plt.plot(xf, yf, 'g.')
xf, yf = maxima_pendiente(f_z_3, x05, y05, eps)
plt.plot(xf, yf, 'g.')
plt.title('Máxima pendiente')
plt.show()
# --------------------------------------------------Ejercicio 7 (extra)-------------------------------------------------
"""
Implementar el método del gradiente conjugado, y aplicarlo a la función de apartado
2 para los mismos puntos iniciales. Añadir a la representación del apartado 3 las iteraciones
obtenidas. ¿Qué diferencias observas respecto al método de la máxima pendiente?
"""


def gradiente_conjugado_fletcher_reeves(f, x0, y0, prec):
    err = 1
    v_old = gradiente_diferencias_centradas(f, x0, y0)
    norm_v_old = (v_old[0] ** 2 + v_old[1] ** 2) ** (1 / 2)

    def f_2(k):
        return f(x0 + k * v_old[0] / norm_v_old, y0 + k * v_old[1] / norm_v_old)

    k = razon_aurea_maximizar(f_2, -1.5, 1.5, prec)
    x1, y1 = x0 + k * v_old[0] / norm_v_old, y0 + k * v_old[1] / norm_v_old
    plt.plot(x1, y1, 'r.')
    plt.arrow(x0, y0, x1 - x0, y1 - y0, length_includes_head=True, head_width=0.1, head_length=0.05, color='k')
    err = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)
    x0, y0 = x1, y1
    v_mixed = np.array(v_old)
    cont = 1
    while err > prec:
        v2 = np.array(gradiente_diferencias_centradas(f, x0, y0))
        norm_v2 = (v2[0] ** 2 + v2[1] ** 2) ** (1 / 2)
        beta = norm_v2 ** 2 / norm_v_old ** 2  # Formula de Fletcher-Reeves
        v = v2 + beta * v_mixed
        norm_v = (v[0] ** 2 + v[1] ** 2) ** (1 / 2)

        def f_2(k):
            return f(x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v)

        k = razon_aurea_maximizar(f_2, -1.5, 1.5,
                                  prec)  # Los valores de x1 y x4 hay que "tunearlos" dependiendo de la estimación inicial.
        x1, y1 = x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v  # Máximo en la dirección
        plt.plot(x1, y1, 'r.')
        plt.arrow(x0, y0, x1 - x0, y1 - y0, length_includes_head=True, head_width=0.1, head_length=0.05, color='k')
        err = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)
        norm_v_old = norm_v2
        v_mixed = v
        x0, y0 = x1, y1
        cont += 1
    print('Se han necesitado', str(cont), 'iteraciones para llegar al máximo:')
    print('x=', x1, ' , y=', y1, sep="")
    print()
    return x1, y1


# Utilizamos la misma estimación que en el apartado 5
x0, y0 = 1.5, -0.5
# Repetimos el plot de densidad
plt.imshow(f_z(x_prima, y_prima).T, cmap='terrain', extent=(-3, 3, -3, 3), origin='lower')
plt.plot(x0, y0, 'b.')
plt.colorbar()
xf, yf = gradiente_conjugado_fletcher_reeves(f_z_3, x0, y0, eps)
plt.plot(xf, yf, 'g.')
plt.title('Gradiente conjugado Fletcher-Reeves')
plt.show()
print(xf, yf)

"""
Hemos necesitado una iteración más que en el caso del método de la máxima pendiente para el
mismo punto inicial. Veamos ahora qué pasa para los mismo puntos iniciales que en el apartado 6.
"""
# Repetimos el apartado anterior con puntos en torno al origen
# para alcanzar el resto de máximos relativos
x0, y0 = 0, 0
x01, y01 = 0.1, 0
x02, y02 = -0.1, 0
x03, y03 = 0, 0.1
x04, y04 = 0, -0.1
x05, y05 = -0.1, 0.1
plt.imshow(f_z(x_prima, y_prima).T, cmap='terrain', extent=(-3, 3, -3, 3), origin='lower')
plt.plot(x0, y0, 'b.')
plt.plot(x01, y01, 'b.')
plt.plot(x02, y02, 'b.')
plt.plot(x03, y03, 'b.')
plt.plot(x04, y04, 'b.')
plt.plot(x05, y05, 'b.')
plt.colorbar()
xf, yf = gradiente_conjugado_fletcher_reeves(f_z_3, x0, y0, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_fletcher_reeves(f_z_3, x01, y01, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_fletcher_reeves(f_z_3, x02, y02, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_fletcher_reeves(f_z_3, x03, y03, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_fletcher_reeves(f_z_3, x04, y04, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_fletcher_reeves(f_z_3, x05, y05, eps)
plt.plot(xf, yf, 'g.')
plt.title('Gradiente conjugado Fletcher-Reeves')
plt.show()


def gradiente_conjugado_polak_ribiere(f, x0, y0, prec):
    err = 1
    v_old = gradiente_diferencias_centradas(f, x0, y0)
    norm_v_old = (v_old[0] ** 2 + v_old[1] ** 2) ** (1 / 2)

    def f_2(k):
        return f(x0 + k * v_old[0] / norm_v_old, y0 + k * v_old[1] / norm_v_old)

    k = razon_aurea_maximizar(f_2, -1.5, 1.5, prec)
    x1, y1 = x0 + k * v_old[0] / norm_v_old, y0 + k * v_old[1] / norm_v_old
    plt.plot(x1, y1, 'r.')
    plt.arrow(x0, y0, x1 - x0, y1 - y0, length_includes_head=True, head_width=0.1, head_length=0.05, color='k')
    err = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)
    x0, y0 = x1, y1
    v_mixed = np.array(v_old)
    cont = 1
    while err > prec:
        v2 = np.array(gradiente_diferencias_centradas(f, x0, y0))
        norm_v2 = (v2[0] ** 2 + v2[1] ** 2) ** (1 / 2)
        beta = np.dot(v2, v2 - v_old) / norm_v_old ** 2  # Fórmula de Polak-Ribiere
        v = v2 + beta * v_mixed
        norm_v = (v[0] ** 2 + v[1] ** 2) ** (1 / 2)

        def f_2(k):
            return f(x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v)

        k = razon_aurea_maximizar(f_2, -1.5, 1.5,
                                  prec)  # Los valores de x1 y x4 hay que "tunearlos" dependiendo de la estimación inicial.
        x1, y1 = x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v  # Máximo en la dirección
        plt.plot(x1, y1, 'r.')
        plt.arrow(x0, y0, x1 - x0, y1 - y0, length_includes_head=True, head_width=0.1, head_length=0.05, color='k')
        err = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)
        norm_v_old = norm_v2
        v_mixed = v
        x0, y0 = x1, y1
        cont += 1
    print('Se han necesitado', str(cont), 'iteraciones para llegar al máximo:')
    print('x=', x1, ' , y=', y1, sep="")
    print()
    return x1, y1


# Utilizamos la misma estimación que en el apartado 5
x0, y0 = 1.5, -0.5
# Repetimos el plot de densidad
plt.imshow(f_z(x_prima, y_prima).T, cmap='terrain', extent=(-3, 3, -3, 3), origin='lower')
plt.plot(x0, y0, 'b.')
plt.colorbar()
xf, yf = gradiente_conjugado_polak_ribiere(f_z_3, x0, y0, eps)
plt.plot(xf, yf, 'g.')
plt.title('Gradiente conjugado Polak-Ribière')
plt.show()
print('Gradiente conjugado Polak-Ribière', xf, yf)

"""
Volvemos a necesitar una iteración más que con el método de la máxima pendiente, ¿qué ocurrirá
al partir de los puntos iniciales del apartado 6?
"""
# Repetimos el apartado anterior con puntos en torno al origen
# para alcanzar el resto de máximos relativos
x0, y0 = 0, 0
x01, y01 = 0.1, 0
x02, y02 = -0.1, 0
x03, y03 = 0, 0.1
x04, y04 = 0, -0.1
x05, y05 = -0.1, 0.1
plt.imshow(f_z(x_prima, y_prima).T, cmap='terrain', extent=(-3, 3, -3, 3), origin='lower')
plt.plot(x0, y0, 'b.')
plt.plot(x01, y01, 'b.')
plt.plot(x02, y02, 'b.')
plt.plot(x03, y03, 'b.')
plt.plot(x04, y04, 'b.')
plt.plot(x05, y05, 'b.')
plt.colorbar()
xf, yf = gradiente_conjugado_polak_ribiere(f_z_3, x0, y0, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_polak_ribiere(f_z_3, x01, y01, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_polak_ribiere(f_z_3, x02, y02, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_polak_ribiere(f_z_3, x03, y03, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_polak_ribiere(f_z_3, x04, y04, eps)
plt.plot(xf, yf, 'g.')
xf, yf = gradiente_conjugado_polak_ribiere(f_z_3, x05, y05, eps)
plt.plot(xf, yf, 'g.')
plt.title('Gradiente conjugado Polak-Ribière')
plt.show()
"""
En todos los casos se mejora el número de iteraciones del método de la máxima pendiente, excepto
en un caso en el que se iguala la convergencia. Este es un buen ejemplo de que dependiendo del
problema, un método puede ser adecuado y otro no.
"""
