"""
Practica 3b: Ecuación de Poisson en 3 dimensiones con aproximación
adiabática
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Ejercicio 1: En esta práctica nos vamos a centrar en resolver la ecuación de Poisson en una caja
tridimensional de lado 𝐿 = 1 m, con un espaciado de malla 𝑎 = 1 cm, en la que se han
colocado de forma alternada bloques de cargas positivas y negativas en las esquinas (veáse
esquema en la figura superior), que representan la densidad de carga 𝜌 del problema. Dichos
bloques los vamos a modelar como cubos de lado 25𝑎 y carga constante de valor ±8, 85×10−12
C m−3.
Calcular el potencial eléctrico en la malla con una precisión de 5 × 10−6 haciendo uso del
método de la relajación.
"""
# Definimos constantes
L = 1.0  # longitud de la caja en metros
M = 100  # número de cubos en cada lado
a = L / M  # tamaño de los cubos de la malla

rho0 = 8.85e-12  # densidad de carga
epsilon0 = 8.85e-12  # permitividad del vacío
eps = 5e-6  # precisión pedida

# En primer lugar creamos los arrays de phi y rho para aplicar el metodo de Jacobi
phi = np.zeros([M + 1, M + 1, M + 1], float)
phiprime = np.zeros([M + 1, M + 1, M + 1], float)

# Con slicinf definimos rho
rho = np.zeros([M + 1, M + 1, M + 1], float)
rho[0:26, 0:26, 0:26] = rho0
rho[-26:, 0:26, 0:26] = -rho0
rho[0:26, -26:, 0:26] = -rho0
rho[-26:, -26:, 0:26] = rho0
rho[0:26, 0:26, -26:] = -rho0
rho[-26:, 0:26, -26:] = rho0
rho[0:26, -26:, -26:] = rho0
rho[-26:, -26:, -26:] = -rho0

# Aplicamos el metodo de Jacobi
delta = 1.0
while delta > eps:
    # Calculamos los nuevos valores del potencial
    phiprime[1:M, 1:M, 1:M] = (phi[0:M - 1, 1:M, 1:M] + phi[2:M + 1, 1:M, 1:M] + phi[1:M, 0:M - 1, 1:M] + phi[1:M,
                                                                                                          2:M + 1,
                                                                                                          1:M] +
                               phi[1:M, 1:M, 0:M - 1] + phi[1:M, 1:M, 2:M + 1]) / 6 + rho[1:M, 1:M, 1:M] * a * a / (
                                          6 * epsilon0)
    # Estimamos el error como el máximo del cambio entre dos estimaciones consecutivas
    delta = np.max(abs(phi - phiprime))
    # Intercambiamos los dos arrays de phi
    phi, phiprime = phiprime, phi

"""
Ejercicio 2: Representar el resultado obtenido en los planos 𝑦 = 35 cm e 𝑦 = 65 cm con imshow,
y el patrón de color seismic.
"""
# Realizamos los plots, primero en el plano y=35cm
plt.imshow(phi[:, 35, :], cmap='seismic')
plt.title('Potencial en el plano y=35 cm')
plt.colorbar()
plt.show()

# Y ahora para y=65 cm
plt.imshow(phi[:, -35, :], cmap='seismic')
plt.title('Potencial en el plano y=65 cm')
plt.colorbar()
plt.show()

# Por simetría del problema y de los planos, el resultado obtenido es el contrario en las dos
# representaciones de densidad

"""
Ejercicio 3: Si 𝜌 varía muy lentamente en el tiempo, podemos considerar un límite cuasiestático en
el que calculemos 𝜙(𝑡) haciendo uso únicamente de la primera ley de Maxwell con distribución
de carga 𝜌(𝑡).
Calcular el potencial en el caso de una variación adiabática de las cargas de la forma 𝜌(𝑡) =
𝜌0 cos(2𝜋𝑡) (𝜌0 es el valor de la carga para cada bloque utilizado en el apartado 1), para 21
instantes de tiempos equiespaciados entre 𝑡 = 0 s y 𝑡 = 1 s.
"""

# Volvemos a crear los arrays, definimos los tiempos, y creamos un array (potentials) para almacenar
# los resultados obtenidos para el potencial en cada tiempo
tp = np.linspace(0, 1, 21)
phi = np.zeros([M + 1, M + 1, M + 1], float)
phiprime = np.zeros([M + 1, M + 1, M + 1], float)
rho = np.zeros([M + 1, M + 1, M + 1], float)
potentials = np.zeros([21, M + 1, M + 1, M + 1], float)

cont = 0  # Con esta variable pasaremos al siguiente valor del primer índice de potentials
# tras haber obtenido el resultado para la iteración presente
# Procedemos igual que en el primer apartado, pero definiendo la densidad de carga en cada tiempo
for t in tp:
    fact = np.cos(2 * np.pi * t)
    rho[0:26, 0:26, 0:26] = rho0 * fact
    rho[-26:, 0:26, 0:26] = -rho0 * fact
    rho[0:26, -26:, 0:26] = -rho0 * fact
    rho[-26:, -26:, 0:26] = rho0 * fact
    rho[0:26, 0:26, -26:] = -rho0 * fact
    rho[-26:, 0:26, -26:] = rho0 * fact
    rho[0:26, -26:, -26:] = rho0 * fact
    rho[-26:, -26:, -26:] = -rho0 * fact
    # bucle
    delta = 1.0
    while delta > eps:
        # Calculamos los nuevos valores del potencial
        phiprime[1:M, 1:M, 1:M] = ((phi[0:M - 1, 1:M, 1:M] + phi[2:M + 1, 1:M, 1:M] + phi[1:M, 0:M - 1, 1:M] +
                                    phi[1:M, 2:M + 1, 1:M] + phi[1:M, 1:M, 0:M - 1] + phi[1:M, 1:M, 2:M + 1]) / 6 +
                                   rho[1:M, 1: M, 1: M] * a * a / (6 * epsilon0))
        # Estimamos el error como el máximo del cambio entre dos estimaciones consecutivas
        delta = np.max(abs(phi - phiprime))
        # Intercambiamos los dos arrays de phi
        phi, phiprime = phiprime, phi

    # Guardamos el resultados en potentials
    potentials[cont] = phi
    # Aumentamos en 1 cont para guardar apropiadamente el cálculo para el siguiente tiempo
    cont += 1

"""
Ejercicio 4: Representar el potencial eléctrico en el plano 𝑦 = 35 cm con imshow, y el patrón
de color seismic, para los resultados obtenidos en primer, séptimo, noveno, decimoséptimo,
decimonoveno y último lugar. ¿Por qué el resultado final difiere del inicial?
"""
# Realizamos las representaciones en el plano y=35cm en los tiempos pedidos
plt.imshow(potentials[0, :, 35, :], cmap='seismic')
plt.colorbar()
plt.show()
plt.imshow(potentials[6, :, 35, :], cmap='seismic')
plt.colorbar()
plt.show()
plt.imshow(potentials[8, :, 35, :], cmap='seismic')
plt.colorbar()
plt.show()
plt.imshow(potentials[16, :, 35, :], cmap='seismic')
plt.colorbar()
plt.show()
plt.imshow(potentials[18, :, 35, :], cmap='seismic')
plt.colorbar()
plt.show()
plt.imshow(potentials[-1, :, 35, :], cmap='seismic')
plt.colorbar()
plt.show()
