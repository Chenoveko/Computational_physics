import numpy as np

#--------------cambios de coordenadas-------------------

def polares_2_cartesianas(r:'Radio en metros', theta:'angulo en radianes') -> 'Coordenadas cartesianas':
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x,y

def cartesianas_2_polares(x:'Posicion eje abscisas',y:'Posicion eje ordenadas') -> 'Coordenadas polares':
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    return r,theta

def esfericas_2_cartesianas(r:'Radio en metros',theta:'Colatitud en radianes [0,π]',phi:'Azimut en radianes [0,2π]') -> 'Coordenadas cartesianas':
    x = r * np.sin(theta)*np.cos(phi)
    y = r * np.sin(theta)*np.sin(phi)
    z = r * np.cos(theta)
    return x,y,z

def cilindricas_2_cartesianas(rho:'Radio en metros',phi:'Azimut en radianes [0,2π]',z:'Altura en metros') -> 'Coordenadas cartesianas':
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = z
    return x,y,z

def grados_2_radianes(x:'Valor en grados [0,360]') -> 'Valor en radianes':
    return x * np.pi / 180

def radianes_2_grados(x:'Valor en radianes [0,2π]') -> 'Valor en grados':
    return x * 180 / np.pi
