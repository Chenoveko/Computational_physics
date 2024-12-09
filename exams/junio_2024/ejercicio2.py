"""
Ejercicio 2: la edad evolución del universo
"""
import numpy as np
from ode.ode import runge_kutta_4_system_method

"""
- 1. H0 estudiar cual va a ser la evolucion del universo es los proximos 30 billones de a˜nos y explicar si la expansion
 del universo se acelerara o desacelerara en este periodo de tiempo
- 2. A partir de los resultados para la velocidad del factor de escala, calcular de forma numerica su aceleracion
y compararla con la evaluacion directa del lado derecho de la ecuacion (2). ¿Cual es la mayor diferencia
entre ambos resultados y cuales son los puntos donde ambas derivadas difieren mas? ¿Por que?
"""

# Constantes
a0 = 1
omega_m = 0.315
omega_lambda = 0.6847
omega_r = 9.4e-5
H_0 = 67.4e-19*60*60*24*365*1e9
omega_kappa = 1 - omega_m - omega_lambda - omega_r

def aceleracion(r):
    factor_escala = r[0]
    velocidad = r[1]
    f_factor_escala = velocidad
    f_velocidad = -H_0**2/2 * omega_m * (2*omega_r*factor_escala**-4 + omega_m*factor_escala**-3 - 2*omega_lambda)/(H_0**-2*velocidad**2-omega_r*factor_escala**-2-omega_lambda*factor_escala**2-omega_kappa)
    return np.array([f_factor_escala, f_velocidad], float)

tp,factor_escala, velocidad = runge_kutta_4_system_method(aceleracion,[0,0],0,30,100)

