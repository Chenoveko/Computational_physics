"""
Ejemplo 8.4: solución de la ecuación de ondas con el método FTCS
"""

# Constantes
L = 1.0  # constantes de la cuerda
v = 100.0
d = 0.1
C = 1.0
sigma = 0.3

N = 100  # intevarlo espacial
a = L / N
h = 1e-6  # paso temporal


# función que nos describe la evolución espacial de la cuerda
def f(y):
    res = np.empty(N + 1, float)
    res[1:N] = (y[0:N - 1] + y[2:N + 1] - 2 * y[1:N]) * v * v / a / a
    res[0] = res[N] = 0.0
    return res


# Creamos el array inicial de $y$ y de $z$
x = np.linspace(0.0, L, N + 1)
psi = np.zeros(N + 1, float)
dpsi = C * x * (L - x) * np.exp(-(x - d) ** 2 / (2 * sigma * sigma)) / (L * L)
# pintamos las condiciones iniciales
plt.plot(x, psi)
plt.plot(x, dpsi)
plt.show()