"""
Ejercicio 1: la edad del universo
"""
import numpy as np
import matplotlib.pyplot as plt

"""
- 1. Representar en primer lugar el integrando de la integral correspondiente al factor de escala
- 2. ¿Que metodo de integracion consideras mas apropiado para integrar dicha funcion? ¿Por que? usar la
ecuacion (1) para, mediante una integral, calcular el valor de t0 y por tanto la edad el universo.
"""

# Constantes
omega_m = 0.315
omega_lambda = 0.6847
omega_r = 9.4e-5
H_0 = 67.4e-19*60*60*24*365*1e9
omega_kappa = 1 - omega_m - omega_lambda - omega_r


# Representar en primer lugar el integrando de la integral correspondiente al factor de escala
def integrando_factor_escala(a):
    return 1/(a*(H_0*np.sqrt(omega_r*a**-4 + omega_m*a**-3 + omega_kappa*a**-2 + omega_lambda)))

a = np.linspace(0,1,1000)
plt.plot(a,integrando_factor_escala(a))
plt.title("Integrando de la integral correspondiente al factor de escala")
plt.xlabel("a")
plt.show()

"""
¿Que metodo de integracion consideras mas apropiado para integrar dicha funcion? ¿Por que?:
cuadratura gaussiana porque la función es suave
"""

"""
Usar la ecuacion (1) para, mediante una integral, calcular el valor de t0 y por tanto la edad el universo.
"""
from integrals_derivatives.integrals import gaussian_quadrature

t0 = gaussian_quadrature(integrando_factor_escala,0,1, 1000)

print('Edad del universo = ',t0)





