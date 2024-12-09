"""
Practica 3a: Propagación ondas EM en el vacío
"""
import numpy as np
import matplotlib.pyplot as plt

"""
Ejercicio 1: Como condiciones iniciales, vamos a considerar
    𝐸𝑥(𝑧, 𝑡 = 0) = 𝐸0 sin(2𝜋𝑧) , 𝐻𝑦(𝑧, 𝑡 = 0) = 𝐻0 sin(2𝜋𝑧),
donde 𝐸0 = 10 Vm−1 y 𝐻0 = 10 Am−1, consideraremos 𝑧 ∈ [0, 2].
Representar las condiciones iniciales en 3 dimensiones, tomando 200 puntos equiespaciados.
Añadir la leyenda a la representación para diferenciar los dos campos, y una línea negra para
destacar el eje 𝑍.
"""

# Número de puntos del espaciado en z
n = 200

# Definimos campo eléctrico inicial
E0 = 10


def E_inicial(z):
    return E0 * np.sin(2 * np.pi * z)


z = np.linspace(0, 2, n)
Ex = E_inicial(z)

# Definimos campo magnético inicial
H0 = 10


def H_inicial(z):
    return H0 * np.sin(2 * np.pi * z)


Hy = H_inicial(z)
cer = np.zeros(n)

# Realizamos el plot siguiendo las ayudas
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(z, cer, Ex, label='Campo Eléctrico')
ax.plot(z, Hy, cer, label='Campo Magnético')
ax.plot([0, 2], [0, 0], [0, 0], 'k-')
plt.title('Onda EM inicial')
plt.legend()
plt.show()

"""
Ejercicio 2: A partir de las ecuaciones discretizadas definir una función que tome los valores de 𝐸𝑥 y 𝐻𝑦 en el 
eje 𝑍 en un tiempo 𝑡, y devuelva los valores de 𝐸𝑥 y 𝐻𝑦 en un tiempo 𝑡 + ℎ𝑡
"""
# Implementamos la evolución temporal

beta = 0.25

def maxwell(E,H):
    En=np.zeros(n)
    Hn=np.zeros(n)
    En[1:n-1]=E[1:n-1]-beta*(H[2:n]-H[0:n-2]) # Con el slicing calculamos Ex y Hy de forma rápida
    Hn[1:n-1]=H[1:n-1]-beta*(E[2:n]-E[0:n-2]) # y compacta
    En[0]=E[0]-beta*(H[1]-H[n-2]) # Para z=0 y z=2, calculamos los nuevos campos de forma análoga
    En[-1]=En[0] # pero haciendo uso de la condición de contorno periódica
    Hn[0]=H[0]-beta*(E[1]-E[n-2])
    Hn[-1]=Hn[0]
    return En, Hn

"""
Ejercicio 3: Realizar 20 iteraciones sucesivas de la función definida en el apartado anterior partiendo
de la condición inicial representada en el apartado 1. Añadir el resultado obtenido a
la representación del primer apartado. ¿Qué observas?
"""
# Creamos el plot en 3 dimensiones y representamos la condición inicial
Ex=E_inicial(z)
Hy=H_inicial(z)

fig,ax=plt.subplots(subplot_kw={"projection":"3d"})
ax.plot(z,cer,Ex,label='Campo Eléctrico')
ax.plot(z,Hy,cer,label='Campo Magnético')
ax.plot([0,2],[0,0],[0,0],'k-')
plt.legend()

# Calculamos 20 iteraciones de la evolucion temporal
for k in range(20):
    Ex,Hy=maxwell(Ex,Hy)

# Añadimos los Ex y Hy obtenidos al plot
ax.plot(z,cer,Ex,label='Campo Eléctrico tras iterar')
ax.plot(z,Hy,cer,label='Campo Magnético tras iterar')
plt.legend()
plt.show()
# Observamos cómo las ondas se desplazan con velocidad constante en la dirección creciente de Z
# sin cambiar su amplitud ni forma