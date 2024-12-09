import numpy as np
import matplotlib.pyplot as plt
from fourier_analysis import *

"""
Ejemplo 6.1: la transformada de Fourier discreta
Escribir un programa que calcule la transformada de Fourier discreta de una muestra real de datos
contenidos en una array.
Cargar el archivo “pitch.txt” en la carpeta de datos del tema 6 y representar los coeficientes de
Fourier de esa función.
"""

pitch = np.loadtxt('data/pitch.txt', float)

plt.plot(pitch)
plt.title('Pitch data from pitch.txt')
plt.show()

coef_pitch = discrete_fourier_transform(pitch)  # Los coeficientes son números complejos

plt.plot(abs(coef_pitch))
plt.title('Fourier coefficients from pitch.txt')
plt.show()

"""
Ejemplo 6.2: transformada de Fourier de funciones simples
Para las siguientes funciones periódicas, calcular los coeficientes de la DFT usando N = 100 puntos
igualmente separados y hacer un gráfico de las amplitudes.
1. Un ciclo de una onda cuadrada con amplitud 1.
2. Una onda dentada con yn = n.
3. Un onda sinusoidal modulada yn = sin ( n =N) sin (20 n =N).
"""


# onda cuadrada de amplitud 1

def square_wave(x):
    if x < 0.5:
        return 1
    else:
        return -1


# onda sinusoidal modulada

def sin_mod(x):
    return np.sin(x * np.pi) * np.sin(20 * np.pi * x)


# Representación señal cuadrada y coeficientes de Fourier

N = 100
x1 = np.linspace(0, 1, N)
y1 = list(map(square_wave, x1))
coef_square = discrete_fourier_transform(y1)

plt.plot(x1, y1)
plt.title('Square wave')
plt.show()

plt.plot(abs(coef_square))
plt.title('Fourier coefficients from square wave')
plt.show()

# Representación onda dentada y coeficientes de Fourier

y2 = np.arange(N)
coef_dentada = discrete_fourier_transform(y2)

plt.plot(x1, y2)
plt.title('Onda dentada')
plt.show()

plt.plot(abs(coef_dentada))
plt.title('Fourier coefficients from onda dentada')
plt.show()

# Representación onda modulada y coeficientes de Fourier

y3 = sin_mod(x1)
coef_mod = discrete_fourier_transform(y3)

plt.plot(x1, y3)
plt.title('Onda modulada')
plt.show()

plt.plot(abs(coef_mod))
plt.title('Fourier coefficients from onda modulada')
plt.show()

"""
Ejemplo 6.3: detectando la periodicidad
"""

sunspots = np.loadtxt('data/sunspots.txt', float)
N = len(sunspots)

mes, num_manchas = sunspots[:, 0], sunspots[:, 1]

plt.plot(mes, num_manchas)
plt.title('Sunspots data from sunspots.txt')
plt.show()

coef_sunspots = discrete_fourier_transform(num_manchas)

plt.plot(abs(coef_sunspots) ** 2)
plt.title('Fourier coefficients from sunspots.txt')
plt.show()

k = 24  # frecuencia dominante
print('frecuencia', N / k)

"""
Ejemplo 6.5: transformada de Fourier de instrumentos musicales
"""

datapiano = np.loadtxt('data/piano.txt', float)
datatrumpet = np.loadtxt('data/trumpet.txt', float)

plt.plot(datapiano)
plt.title('Piano data from piano.txt')
plt.xlim(0, 10000)
plt.show()

coef_piano = np.fft.rfft(datapiano)

plt.plot(abs(coef_piano))
plt.title('Fourier coefficients from piano.txt')
plt.show()

plt.plot(datatrumpet)
plt.title('Trumpet data from trumpet.txt')
plt.xlim(0, 10000)
plt.show()

coef_trumpet = np.fft.rfft(datatrumpet)

plt.plot(abs(coef_trumpet))
plt.title('Fourier coefficients from trumpet.txt')
plt.show()

"""
Ejemplo 6.6: filtrado de Fourier y suavizado de funciones
"""

dow = np.loadtxt('data/dow.txt')

plt.plot(dow)
plt.title('Dow data from dow.txt')
plt.show()

coef_dow = np.fft.rfft(dow)
plt.plot(abs(coef_dow))
plt.title('Fourier coefficients from dow.txt')
plt.show()

"""
A continuación poner a cero todos los elementos del array excepto los que corresponden al
primer 10% (es decir, poner el último 90% de los coeficientes a cero y mantener sólo los valores
del primer 10 %)
"""
n = int(0.1 * len(coef_dow))
coef_dow[n:] = 0.0
dow_suavizado = np.fft.irfft(coef_dow)
plt.plot(dow, label='original')
plt.plot(dow_suavizado, label='suavizado')
plt.title('Suavizado de dow.txt')
plt.legend()
plt.show()

"""
Ejercicio 6.1: la onda cuadrada y el efecto Gibbs
"""
# Representación

N = 1000
x = np.linspace(0, 1, N)
y = list(map(square_wave, x))
plt.plot(x, y)
plt.title('Square wave')
plt.show()

# transformada de Fourier

coef_square = np.fft.rfft(y)

plt.plot(abs(coef_square))
plt.title('Fourier coefficients from square wave')
plt.show()

# suavizamos la señal poniendo a cero todos los coeficientes con n>10
coef_square[10:] = 0.0
y_suavizado = np.fft.irfft(coef_square)

plt.plot(x, y, label='original')
plt.plot(x, y_suavizado, label='suavizado')
plt.title('Suavizado de la onda cuadrada')
plt.legend()
plt.show()

"""
Ejemplo 6.5: transformada de coseno y seno rápidas
Escribir un programa que calcula la FCT y la FST directa e inversa de tipo II
"""


def discrete_cosen_transform(y):
    N = len(y)
    y2 = np.empty(2 * N, float)

    for n in range(N):
        y2[n] = y[n]
        y2[2 * N - 1 - n] = y[n]

    c = np.fft.rfft(y2)
    phi = np.exp(-1j * np.pi * np.arange(N) / (2 * N))

    return np.real(phi * c[:N])


def inverse_discrete_cosen_transform(a):
    N = len(a)

    c = np.empty(N, complex)

    phi = np.exp(1j * np.pi * np.arange(N) / (2 * N))
    c[:N] = phi * a
    return np.fft.irfft(c)[:N]
