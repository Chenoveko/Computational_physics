import numpy as np
import matplotlib.pyplot as pyplot

def forward_derivative(f: 'function', x: 'points') -> 'Values of the forward derivative':
    h = 1e-8
    return (f(x+h) - f(x))/h

def backward_derivative(f: 'function', x: 'points') -> 'Values of the backward derivative':
    h = 1e-8
    return (f(x) - f(x-h))/h

def centered_derivative(f: 'function', x: 'points') -> 'Values of the centered derivative':
    h = 1e-5
    return (f(x+h/2) - f(x-h/2))/h

# use case -> Example 4.11
y = np.linspace(-1,1,100)

def f(x):
    return 1 +1/4*np.tanh(4*x)

def f_primed(x):
    return 1/np.cosh(4*x)**2

pyplot.plot(y,f(y))
pyplot.plot(y,f_primed(y))
pyplot.plot(y,forward_derivative(f,y))
pyplot.plot(y,backward_derivative(f,y))
pyplot.plot(y,centered_derivative(f,y))
pyplot.show()


