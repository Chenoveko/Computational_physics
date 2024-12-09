import numpy as np
import matplotlib.pyplot as plt


# ------------------------------Ejercicio 1------------------------------
def distribucion_masa(x, y, z):
    a = 1
    b = 0.5
    rho_0 = 1
    phi = np.arctan2(y / b, x / a)
    if (x ** 2 + 4 * y ** 2) <= 4 and (z >= -1 and z <= 1):
        if phi < 0:
            phi += 2 * np.pi
            return rho_0 * phi ** 2 * (1 - z ** 2)
        else:
            return rho_0 * phi ** 2 * (1 - z ** 2)
    else:
        return 0


N = 1000
x = np.linspace(-2, 2, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

distribucion_masa_vectorizada = np.vectorize(distribucion_masa)

mi_distribucion = distribucion_masa_vectorizada(X, Y, 0)

plt.imshow(mi_distribucion, origin='lower', cmap='Greens', extent=(-2, 2, -1, 1))
plt.colorbar(label="Distribucion masa")  # Agregar etiqueta a la barra de color
plt.title("Distribución de masa plano z = 0")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

# ------------------------Ejercicio 2-----------------------------------

"""
utilizo cuadratura gaussiana porque la función es suave y obtengo mayor precision
"""

from gaussxw import gaussxwab


def momento_inercia_z(x0, y0):
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I += w_x[i] * w_y[j] * w_z[k] * ((x[i] - x0) ** 2 + (y[j] - y0) ** 2) * distribucion_masa(x[i], y[j],
                                                                                                          z[k])
    return I


N = 150
x = np.linspace(-2, 2, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

mi_momento = momento_inercia_z(X, Y)

plt.imshow(mi_momento, origin='lower', cmap='Greens', extent=(-2, 2, -1, 1))
plt.colorbar(label="Momento de inercia plano XY")  # Agregar etiqueta a la barra de color
plt.title("Momento de inercia")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()


# ---------------------------------Ejercicio 3------------------------

def momento_inercia_xx():
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I += w_x[i] * w_y[j] * w_z[k] * (y[j] ** 2 + z[k] ** 2) * distribucion_masa(x[i], y[j], z[k])
    return I


def momento_inercia_yy():
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I += w_x[i] * w_y[j] * w_z[k] * (x[i] ** 2 + z[k] ** 2) * distribucion_masa(x[i], y[j], z[k])
    return I


def momento_inercia_zz():
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I += w_x[i] * w_y[j] * w_z[k] * (x[i] ** 2 + y[j] ** 2) * distribucion_masa(x[i], y[j], z[k])
    return I


def momento_inercia_xy():
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I += w_x[i] * w_y[j] * w_z[k] * x[i] * y[j] * distribucion_masa(x[i], y[j], z[k])
    return -1 * I


def momento_inercia_xz():
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I += w_x[i] * w_y[j] * w_z[k] * x[i] * z[k] * distribucion_masa(x[i], y[j], z[k])
    return -1 * I


def momento_inercia_yz():
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I += w_x[i] * w_y[j] * w_z[k] * y[j] * z[k] * distribucion_masa(x[i], y[j], z[k])
    return -1 * I


tensor_inercia = np.zeros((3, 3))

tensor_inercia[0, 0] = momento_inercia_xx()
tensor_inercia[1, 1] = momento_inercia_yy()
tensor_inercia[2, 2] = momento_inercia_zz()
tensor_inercia[0, 1] = tensor_inercia[1, 0] = momento_inercia_xy()
tensor_inercia[1, 2] = tensor_inercia[2, 1] = momento_inercia_yz()
tensor_inercia[2, 0] = tensor_inercia[0, 2] = momento_inercia_xz()
print(tensor_inercia)
# -------------------------------Ejercicio 4-------------------------
L = np.array((2, 2, 2))
print(np.linalg.solve(tensor_inercia, L))


def LU_factorization(A: 'Matrix to factorize') -> 'Lower Upper factorization of the matrix A':
    """
    - La eliminación gaussiana con pivote es un metodo efectivo y rápido.
    - Sin embargo, en muchas ocasiones en física queremos resolver el sistema: A · X = V, para distintos valores de V.
    """
    N = len(A)
    # Inicializamos nuestras dos matrices.
    L = np.zeros([N, N], float)  # L tiene que ser una matriz triangular inferior.
    U = np.copy(A)  # U será la matriz A convertida en triangular superior
    # Creamos la matrix L
    for m in range(N):
        L[m:N, m] = U[m:N,
                    m]  # para cada iteracción (para cada fila), la columna m de L queda fijada por el valor que tiene U.
        # Convertimos ahora U en una matrix triangular superior.
        # Para ello usamos la eliminación guassiana.
        # 1. Dividimos la fila m por el elemento m,m
        div = U[m, m]
        U[m, :] /= div
        # 2. Sustraemos la fila m a las filas i>m multiplicadas por el elemento i,m
        for i in range(m + 1, N):
            mult = U[i, m]
            U[i, :] -= mult * U[m, :]
    return L, U


def LU_solution(A: 'Coeficients matrix', V: 'Column vector') -> 'Solution of the system using LU method':
    N = len(A)
    L, U = LU_factorization(A)
    # 1. Sustitución hacía adelante. Operando con L.
    y = np.empty(N, float)
    for m in range(N):
        y[m] = V[m]
        for i in range(m):
            y[m] -= L[m, i] * y[i]
        y[m] /= L[m, m]
    # 2. Sustitución hacía atrás. Operando con U.
    x = np.empty(N, float)
    for m in range(N - 1, -1, -1):
        x[m] = y[m]
        for i in range(m + 1, N):
            x[m] -= U[m, i] * x[i]
    return x


print("Velocidad angular sin pivote = ", LU_solution(tensor_inercia, L))


# ------------------------------------Ejercicio 5-------------------------


def centro_masas_x():
    N_cuadraturas = 50
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I1 = 0.0
    I2 = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I1 += w_x[i] * w_y[j] * w_z[k] * x[i] * distribucion_masa(x[i], y[j], 0)
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I2 += w_x[i] * w_y[j] * w_z[k] * distribucion_masa(x[i], y[j], 0)

    return I1 / I2


def centro_masas_y():
    N_cuadraturas = 10
    x, w_x = gaussxwab(N_cuadraturas, -2, 2)
    y, w_y = gaussxwab(N_cuadraturas, -1, 1)
    z, w_z = gaussxwab(N_cuadraturas, -1, 1)
    I1 = 0.0
    I2 = 0.0
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I1 += w_x[i] * w_y[j] * w_z[k] * y[i] * distribucion_masa(x[i], y[j], 0)
    for i in range(N_cuadraturas):
        for j in range(N_cuadraturas):
            for k in range(N_cuadraturas):
                I2 += w_x[i] * w_y[j] * w_z[k] * distribucion_masa(x[i], y[j], 0)

    return I1 / I2


print(centro_masas_x(), centro_masas_y())
print(momento_inercia_z(centro_masas_x(), centro_masas_y()))
v_max= 400
v_min = 100
plt.imshow(mi_momento, origin='lower', cmap='Greens', extent=(-2, 2, -1, 1), vmin=v_min, vmax=v_max)
plt.scatter(centro_masas_x(), centro_masas_y())
plt.colorbar(label="Momento de inercia plano XY")  # Agregar etiqueta a la barra de color
plt.title("Momento de inercia")  # Título del gráfico
plt.xlabel("x (eje de las abscisas)")  # Etiqueta para el eje x
plt.ylabel("y (eje de las ordenadas)")  # Etiqueta para el eje y
plt.show()

"""
El punto donde se minimiza el momento de inercia es en el CM de la distribución,tal y como se puede ver en la gráfica 
de acuerdo al código de colores. Lo cual verifica el teorema de Huygenss-steiner
"""
