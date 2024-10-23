from Funciones import fv1, fx, operacion, operacion2, Runge_Kutta4
import numpy as np
import matplotlib.pyplot as plt
from vpython import sphere, color, curve, vector, rate

#DEF Constante Gravitacional
G = -6.673884e-11

numerocuerpos = 3
ni = 1      #numero de interacciones

Ttot = 3600.    
dt = Ttot/ni    #cantidad de tiempo en que se actualizara el sistema
ti = 0          
es = 30         #Valor de la escala
ca = np.pi/180  #Conversion de angulo a radianes

#Inicializacion de listas necesarias para el sistema, el numero de componentes de la lista la define nc
xic = [0] * numerocuerpos    #Lista de la posicion inicial
xfc = [0] * numerocuerpos    #Lista de la posicion final
vic = [0] * numerocuerpos    #Lista de la velocidad inicial
vfc = [0] * numerocuerpos    #Lista de la velocidad final

#Condiciones iniciales, incializacion de masas 
#CASO 1: todas las masas iguales

mc = [5.97e24, 10e30, 2.16e18]

#incializacion de las posiciones iniciales y finales
#A cada componente de la lista xic y vic se inicializa con el np.array posicion inicial y velocidad
#inicial respectivamente para evitar problemas la posicion final se iguala a la posicion inicial antes
#de comenzar el programa

#Cuerpo 1
vic[0]=np.array([np.sin(np.pi), np.cos(np.pi), 0])
xic[0]=np.array([0., 0., 0.])
xfc[0]=xic[0]

#Cuerpo 2
vp=30.288e3-1000
pp=147.10e9+3.84e8
a=282.94 * ca
i=0.00 * ca
vic[1]=np.array([-vp*np.sin(a), vp*np.cos(a), 0])
xic[1]=np.array([pp*np.cos(a)*np.cos(i), pp*np.sin(a)*np.cos(i), pp*np.sin(i)])
xfc[1]=xic[1]

#Cuerpo 3
vp=30.288e3
pp=147.10e9
a = 282.94 * ca
i = 0.00 * ca
vic[2]=np.array([-vp*np.sin(a), vp*np.cos(a), 0])
xic[2]=np.array([pp*np.cos(a)*np.cos(i), pp*np.sin(a)*np.cos(i), pp*np.sin(i)])
xfc[2]=xic[2]

# Suponiendo que xic y xfc sean listas con tres componentes para cada cuerpo
# Ejemplo: xic[0] = [x, y, z]
cuerpo1 = sphere(pos=vector(xic[0][0], xic[0][1], xic[0][2]), radius=5e9, color=color.yellow)
cuerpo1.trail = curve(color=cuerpo1.color)

cuerpo2 = sphere(pos=vector(xic[1][0], xic[1][1], xic[1][2]), radius=5e9, color=color.blue)
cuerpo2.trail = curve(color=cuerpo2.color)

cuerpo3 = sphere(pos=vector(xic[2][0], xic[2][1], xic[2][2]), radius=5e9, color=color.orange)
cuerpo3.trail = curve(color=cuerpo3.color)


#Bucle para calcular la posicion de los cuerpos
n=0
while True:
    rate(1e8)
    n=n+1                   
    tf=ti+dt

    #Llama la funcion Ruge-Kutta para hallar los valores finales de posicion y velociadad                        
    xfc,vfc=Runge_Kutta4(xic, vic, tf, ti, mc, numerocuerpos)

    # Actualización de la nueva posición de los cuerpos
    cuerpo1.pos = vector(xfc[0][0], xfc[0][1], xfc[0][2])
    cuerpo1.trail.append(cuerpo1.pos)

    cuerpo2.pos = vector(xfc[1][0], xfc[1][1], xfc[1][2])
    cuerpo2.trail.append(cuerpo2.pos)

    cuerpo3.pos = vector(xfc[2][0], xfc[2][1], xfc[2][2])
    cuerpo3.trail.append(cuerpo3.pos)

    #Se actualizan los valores iniciales de velocidad, posicion y tiempo                       
    vic=vfc
    xic=xfc
    ti=tf














