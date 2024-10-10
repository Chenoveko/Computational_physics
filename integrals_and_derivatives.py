import numpy as np
from gaussxw import gaussxwab

#--------- Trapezium rule--------------#

def trapezium_rule(f: 'function', a: 'init point', b: 'end point', N: 'number of points') -> 'Value of the integral':
    h = (b-a)/N
    I = (f(a)+f(b))/2
    for i in range(1,N):
        I += f(a+h*i)
    return I*h

# use case for the function x**4-2x+1
def g(x):
    return x**4-2*x+1
print("Trapezium rule ",trapezium_rule(g,0,2,10))


#---------Simpson rule--------------#
def simpson_rule(f: 'function', a: 'init point', b: 'end point', N: 'number of points (must be odd)') -> 'Value of the integral':
    if N % 2 != 0:
        raise Exception("N must be an odd number")
    h = (b - a) / N #step
    I = f(a) + f(b) + 4 * f(b - h)
    for k in range(1, N // 2):
        I += 4 * f(a + (2 * k - 1) * h) + 2 * f(a + 2 * k * h)
    return I * h / 3


# use case for the last function
print("Simpson rule ",simpson_rule(g,0,2,10))


#-------------------------------------------------------Adaptive methods----------------------

#------------------Trapezium rule with adaptive step


def trapezium_rule_adaptive(f: 'function', a: 'init point', b: 'end point', eps_obj: 'error target') -> 'Value of the integral':
    eps = 1.0
    N = 1
    h = (b-a)/N
    I1 = h * (f(a) + f(b))/2
    while eps > eps_obj:
        h /= 2
        I2 = 0
        for k in range(N):
            I2 += f(a + h*(2*k+1))
        I2 = I1/2 + h*I2
        eps = abs(I2-I1)/3
        I1 = I2
        N *= 2
    return I2

def romberg_rule(f: 'function', a: 'init point', b: 'end point', eps_obj: 'error target') -> 'Value of the integral':
    eps = 1.0
    cont = 1
    N = 1
    h = (b-a)/N
    I1 = h / 2 * (f(a) + f(b))
    R1 = np.array([I1], float)
    while eps > eps_obj:
        h /= 2
        I2 = 0
        for k in range(N):
            I2 += f(a + (2 * k + 1) * h)
        I2 = I1 / 2 + h * I2
        cont += 1
        R2 = np.empty(cont, float)
        R2[0] = I2
        for m in range(1, cont):
            eps = abs(R2[m - 1] - R1[m - 1]) / (4 ** m - 1)
            R2[m] = R2[m - 1] + eps
        N *= 2
        I1 = I2
        R1 = R2
    return R1[-1]

def simpson_rule_adaptive(f: 'function', a: 'init point', b: 'end point', eps_obj: 'error target') -> 'Value of the integral':
    eps = 1
    N = 2
    h = (b-a)/N
    S = (f(a)+f(b))/3
    T = f(a+h)*2/3
    I1 = h*(S+2*T)
    while eps > eps_obj:
        h /= 2
        S += T
        T = 0
        for k in range(N):
            T += 2*f(a+(2*k+1)*h)/3
        I2 = h*(S+2*T)
        eps = abs(I2-I1)/15
        I1 = I2
        N *= 2
    return I1

# use case for adaptive methods

def f(x):
    return np.sin(np.sqrt(100*x))**2

print("Trapezium rule adaptive ",trapezium_rule_adaptive(f,0,1,1.0e-6))
print("Simpson rule adaptive ",simpson_rule_adaptive(f,0,1,1.0e-6))
print("Romberg rule ",romberg_rule(f,0,1,1.0e-6))


#-----------------------------Gaussian quadrature-------------------------

def gaussian_quadrature(f: 'function', a: 'init point', b: 'end point', N: 'number of points') -> 'Value of the integral':
    x, w = gaussxwab(N,a,b)
    I = 0.0
    for k in range(N):
        I += (w[k] * f(x[k]))
    return I

print("Gaussian quadrature ",gaussian_quadrature(g,0,2,3))




