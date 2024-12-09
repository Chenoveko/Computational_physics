import numpy as np
from derivatives import *
import matplotlib.pyplot as plt

"""
Ejemplo 4.11: calculando derivadas 
"""


def f(x):
    return 1 + np.tanh(4 * x) / 4


xp = np.linspace(-1, 1, 100)

forward = forward_derivative(f, xp)
backward = backward_derivative(f, xp)
centered = centered_derivative(f, xp)

plt.plot(xp, forward, label='Forward', marker="o", color="black")
plt.plot(xp, backward, label='Backward', marker="o", color="red")
plt.plot(xp, centered, label='Centered', marker="o", color="blue")
plt.legend()
plt.show()
