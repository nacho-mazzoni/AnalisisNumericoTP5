from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#DEF Constante Gravitacional
G = -6.673884e-11

#DEF funcion segunda derivada de r en funcion de t
 #Funcion que encuentra la velocidad final de cada cuerpo en tf, x es un
 #vector que tiene en sus components todas las posiciones de los
 # diferentes objetos, v las velocidades iniciales respectivas a cada
 # objeto, m las masas y n la cantidad de objetos
def fv1(G, x, m, n): 
    vec = np.zeros(3)
    dxi = [vec]*n
    Rmin = 5e7 #radio minimo, si la posicion en la que se encuentra la masa es menor, la velocidad no se vera afectada

    for i in range(0, n):
        for j in range(0,n):
            if i!=j :
                rij = sqrt(np.dot(x[i]-x[j], x[i]-x[j]))
                if rij > Rmin :
                    dxi[i] = dxi[i] + G*m[j]*(x[i]-x[j])/pow(rij, 3)
                else:
                    dxi[i] = dxi[i]
    return dxi

#DEF funcion derivada de la posicion respecto del tiempo, devuelve la posicion del cuerpo
def fx(v):
    return v

#DEF funcion para encontrar los k1, k2, k3 y k4 para el metodo runge-kutta
#Funcion que me realiza las operaciones del metodo
 #runge-kutta, ya que las entradas son listas y no
 #puedo manipularlas diectamente
def operacion(kix, kiv, xio, vio, n, var, h):
    vec = np.zeros(3)
    xip = [vec]*n
    vip = [vec]*n
    kipx = [vec]*n
    kipv = [vec]*n

    for i in range(0, n):
        kipx[i] = h*kix[i]
        kipv[i] = h*kiv[i]
        if var == 1 :
            xip[i] = xio[i] + kipx[i]/2       #Para hallar k2 y k3
            vip[i] = vio[i] + kipv[i]/2
        else :
            xip[i] = xio[i] + kipx[i]          #Para hallar k4
            vip[i] = vio[i] + kipv[i]
    return xip, vip, kipx, kipv

#DEF funcion para hallar los nuevos valores de velocidad y posicion finales
# en el metodo runge-kutta

def operacion2(k1, k2, k3, k4, xio2, n):
    vec = np.zeros(3)
    xip2 = [vec]*n

    for i in range(0, n):
        xip2[i] = xio2[i] + (k1[i]+2*(k2[i]+k3[i])+k4[i])/6
    
    return xip2

#DEF metodo Runge-Kutta

def Runge_Kutta4(xi, vi, tf, ti, m, n):
    #xi, vi vectores con todas las posiciones y velocidades iniciales de los cuerpos
    var = 1     #Variable utilizada en la funcion operacion, para definir la
                #actualizacion para los casos de k2 y k3
    
    dt = tf - ti
    
    k1x = fx(vi)
    k1v = fv1(G, xi, m, n)

    xiv, viv, k1x, k1v = operacion(k1x, k1v, xi, vi, n , var, dt)
    k2x = fx(viv)
    k2v = fv1(G , xiv, m, n)

    xiv, viv, k2x, k2v = operacion(k2x, k2v, xi, vi, n, var, dt)
    k3x = fx(viv)
    k3v = fv1(G, xiv, m, n)

    var = 0
    xiv, viv, k3x, k3v = operacion(k3x, k3v, xi, vi, n, var, dt)
    k4x = fx(viv, )
    k4v = fv1(G, xiv, m, n)

    xiv, viv, k4x, k4v = operacion(k4x, k4v, xi, vi, n, var, dt)

    xiv = operacion2(k1x, k2x, k3x, k4x, xi, n)
    xf = xiv

    viv = operacion2(k1v, k2v, k3v, k4v, vi, n)
    vf = viv

    return xf, vf
