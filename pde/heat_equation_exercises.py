import numpy as np
import matplotlib.pyplot as plt

"""
Ejemplo 8.3: solución para la ecuación del calor con el método FTCS
"""

# Constantes
L = 0.01  # espesor de la barra en metros
D = 4.25e-6  # coeficiente de difusión térmica del acero
N = 100  # número de puntos en x, posición de la sección de espesor L.
a = L / N  # espaciado de la malla espacial
h = 1e-4  # paso en la variable temporal tiempo

k = h * D / a ** 2  # constante para la solución de Euler de la parte temporal

Tlo = 0.0  # temperatura baja en C
Tmid = 20.0  # temperatura inicial del acero en C
Thi = 50.0  # Temperatura alta en C

# Array de tiempos en los que vamos a calcular el perfil de temperaturas
tp = [0.01, 0.1, 0.4, 1.0, 10.0]

# Inicializamos nuestra array de temperaturas
T = np.empty(N + 1, float)  # inicialización
# Fijamos condiciones de contorno
T[0] = Thi  # temperatura de la parte de la barra en contacto con el baño caliente
T[N] = Tlo  # temperatura de la parte de la barra en contacto con el baño frio
T[1:N] = Tmid  # temperatura inicial para el resto de la barra de acero

x = np.linspace(0, 1, 101)  # array con los puntos del grosor del acero

# Loop principal

eps = h / 1000  # precisión para definir un punto de la parte temporal
t = 0  # tiempo inicial
tf = 10 + eps

while t < tf:
    T[1:N] += k * (T[2:N + 1] + T[0:N - 1] - 2 * T[1:N])
    t += h
    # representamos el perfil de temperatura en los tiempos determinados
    for i in tp:
        if abs(t - i) < eps:
            plt.plot(x, T, label="T=%0.2f" % i)

plt.xlabel("x")
plt.ylabel("T")
plt.title("Perfil de temperaturas")
plt.legend()
plt.show()

"""
Ejercicio 8.3: difusión térmica en la corteza terrestre
"""


def t_superficie(t):
    A, B, tau = 10, 12, 365.25
    return A + B * np.sin(2 * np.pi * t / tau)


# Constantes
L = 20  # espesor de la corteza terrestre en metros
D = 0.1  # coeficiente de difusión en m²/día
N = 100  # número de puntos en el espacio
a = L / N  # espaciado de la malla espacial
h = a ** 2 / (2 * D)  # paso temporal en días que garantice la estabilidad

k = h * D / a ** 2  # constante para la solución de FTCS

# Array de tiempos en los que vamos a calcular el perfil de temperaturas
tp = np.arange(0, 365.25 * 10, 365.25 / 4)

# Inicializamos nuestra array de temperaturas
T = np.ones(N + 1, float) * 10
# Fijamos condiciones de contorno
T[N] = 11  # temperatura a 20 m de profundidad
T[0] = t_superficie(0)  # temperatura superficie

x = np.linspace(0, L, N + 1)  # array con los puntos de la profundidad

# Loop principal
eps = h / 2  # usar el paso temporal como precisión
t = 0  # tiempo inicial
tf = 365.25 * 10 + eps  # 10 años en días

while t < tf:
    T[0] = t_superficie(t)  # condición de contorno en la superficie
    T[1:N] += k * (T[2:N + 1] + T[0:N - 1] - 2 * T[1:N])
    t += h
    # representamos el perfil de temperatura en los tiempos determinados
    for i in tp:
        if abs(t - i) < eps:
            plt.plot(x, T, label=f"Día {int(t)}")
            print(T)

plt.xlabel("Profundidad (m)")
plt.ylabel("Temperatura (°C)")
plt.title("Perfil de temperaturas en la corteza terrestre")
plt.legend()
plt.show()