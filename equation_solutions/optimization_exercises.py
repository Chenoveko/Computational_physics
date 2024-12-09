import numpy as np
import matplotlib.pyplot as plt
from optimization import *

"""
Ejemplo 5.20: el potencial de Buckingham
El potencial de Buckingham es una representación aproximada de la energía de interacción entre
los átomos de un sólido ó gas como función de la distancia r entre ellos
"""


def potencial_buckingham(r, sigma=1):
    return (sigma / r) ** 6 - np.exp(-r / sigma)


# Lo representamos para estimar el intervalo
x = np.linspace(0.1, 10, 1000)
plt.ylim(-0.2, 0.1)
plt.title('Potencial de Buckingham')
plt.plot(x, potencial_buckingham(x))
plt.show()

x1 = 0.1
x4 = 10
eps = 1e-6
xmin = golden_ratio_min(potencial_buckingham, x1, x4, eps)
print("El mínimo cae en", xmin, "nm")

plt.plot(x, potencial_buckingham(x))
plt.title('Potencial de Buckingham con mínimo')
plt.plot(xmin, potencial_buckingham(xmin), 'ko')
plt.ylim(-0.2, 0.1)
plt.show()
