
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------Ejercicio 1----------------------------------

desfasaje = np.loadtxt('desfasaje_onda_f.dat')
desfasaje = np.degrees(desfasaje)

elasticidad = np.loadtxt('elasticidad_onda_f.dat')
# estimación inicial
m_rho_0 = 1.7
gamma_rho_0 = 0.25
# constantes
m_pi = 0.13957

x_rho = 0.268

def sigma(s):
    return np.sqrt(1-4*m_pi**2/s)

def t(s,m_rho,gamma_rho):
    return 1/sigma(s) * x_rho * m_rho * gamma_rho / (m_rho **2 - s - 1j *m_rho * gamma_rho)

def S(s,m_rho,gamma_rho):
    return 1 + 2j*sigma(s)*t(s,m_rho,gamma_rho)

def f_desfasaje(s,m_rho,gamma_rho):
    a = S(s,m_rho,gamma_rho)
    return 1/2 * np.arctan(a.imag/a.real)

def f_elasticidad(s, m_rho, gamma_rho):
    return np.abs(S(s,m_rho,gamma_rho))

def xi(m_rho,gamma_rho):
    a,b = 0,0
    for i in range(len(desfasaje)):
        a += (desfasaje[:,1] - f_desfasaje(desfasaje[:,0],m_rho,gamma_rho))**2/ desfasaje[:,2]**2

    for j in range(len(elasticidad)):
        b += (elasticidad[:,1] - f_elasticidad(elasticidad[:,0],m_rho,gamma_rho))**2/ elasticidad[:,2]**2

    return a +b

eps = 1e-3

def gradiente_diferencias_centradas(f, x, y):
    h = 1e-3
    grad = [0, 0]
    grad[0] = (f(x + h / 2, y) - f(x - h / 2, y)) / h
    grad[1] = (f(x, y + h / 2) - f(x, y - h / 2)) / h
    return grad

def razon_aurea_minimizar(f:'funcion',x1:'pto inicial',x4:'pto final',eps:'Precisión objetivo'):
    z = (1 + np.sqrt(5)) / 2  # razón áurea
    x2 = x4 - (x4 - x1) / z
    x3 = x1 + (x4 - x1) / z
    f2 = f(x2)
    f3 = f(x3)
    while x4 - x1 > eps:
        if f2 < f3:
            x4, f4 = x3, f3
            x3, f3 = x2, f2
            x2 = x4 - (x4 - x1) / z
            f2 = f(x2)
        else:
            x1, f1 = x2, f2
            x2, f2 = x3, f3
            x3 = x1 + (x4 - x1) / z
            f3 = f(x3)
    xmin = (x1 + x4) / 2
    return xmin

def maxima_pendiente(f: 'funcion', x0: 'pto x inicial', y0: 'pto y inicial', prec: 'precision objetivo') -> 'Extremo':
    err = 1
    cont = 0
    while err > prec:
        v = gradiente_diferencias_centradas(f, x0, y0)
        norm_v = (v[0] ** 2 + v[1] ** 2) ** (1 / 2)
        def f_2(k):
            return f(x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v)

        k = razon_aurea_minimizar(f_2, -0.5, 0.5,prec)  # Los valores de x1 y x4 hay que "tunearlos" dependiendo de la estimación inicial.
        x1, y1 = x0 + k * v[0] / norm_v, y0 + k * v[1] / norm_v
        err = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** (1 / 2)
        cont += 1
        x0, y0 = x1, y1
    return x1, y1


# m,gamma = maxima_pendiente(xi,m_rho_0,gamma_rho_0,eps)


#------------------------------------------Ejercicio 2-----------------------------------------


raiz_s = np.linspace(2*m_pi,2,1000)

# lo hago con las estimaciones iniciales
plt.plot(raiz_s,f_desfasaje(raiz_s,m_rho_0,gamma_rho_0), label = 'desfasaje')
plt.xlabel("sqrt(s)")
plt.ylabel("desfasaje (grados)")
plt.legend()
plt.show()


plt.plot(raiz_s,f_elasticidad(raiz_s,m_rho_0,gamma_rho_0),label = 'elasticidad')
plt.xlabel("sqrt(s)")
plt.ylabel("elasticidad")
plt.legend()
plt.show()

#------------------------------------------Ejercicio 3-----------------------------------------
N = 500
real_raiz_s = np.linspace(1.4,2,N)
im_raiz_s = np.linspace(-0.25,0,N)

x ,y = np.meshgrid(real_raiz_s,im_raiz_s)

z = abs(t(x+1j*y,m_rho_0,gamma_rho_0))
x0 = 1.41 -0.1j
# Grafico densidad
plt.imshow(z, origin='lower', cmap='YlOrRd', extent=(1.4, 2, -0.25, 0),vmax = 0.1)
plt.colorbar(label="valor absoluto onda parcial")  # Agregar etiqueta a la barra de color
plt.scatter(x0.real,x0.imag)
plt.title("Ejercicio 3")  # Título del gráfico
plt.xlabel("Real")  # Etiqueta para el eje x
plt.ylabel("Imaginario")  # Etiqueta para el eje y
plt.show()

#------------------------------------------Ejercicio 4-----------------------------------------

def centered_derivative(f: 'function', x: 'points') -> 'Values of the centered derivative':
    h = 1e-5
    return (f(x+h/2) - f(x-h/2))/h
def t_inv(s,m_rho,gamma_rho):
    a = 1/sigma(s) * x_rho * m_rho * gamma_rho / (m_rho **2 - s - 1j *m_rho * gamma_rho)
    return 1/a

def jacobiano_numerico(f,pto):
    n = len(pto)
    jacobiano = np.zeros(n)
    for i in range(n):
        jacobiano[i, :] = centered_derivative(f, pto[i])
    return jacobiano


def newton_raphson_multiple_variables(F,x0,err_obj):
    x0=np.array(x0)
    err=1e6
    it=0
    x=np.copy(x0)
    while err>err_obj:
        dx=np.linalg.solve(jacobiano_numerico(t_inv,x0),F(x0))
        err=max(abs(dx))
        x-=dx
        x0=np.copy(x)
        it+=1
    return x0,it

estimacion = [1.41 -0.1j,1.7,0.25]
valor = newton_raphson_multiple_variables(t_inv,estimacion,eps)
