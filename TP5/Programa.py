from Funciones import fv1, fx, operacion, operacion2, Runge_Kutta4
import numpy as np
import matplotlib.pyplot as plt

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

mc = [5,97e24] * numerocuerpos

#incializacion de las posiciones iniciales y finales
#A cada componente de la lista xic y vic se inicializa con el np.array posicion inicial y velocidad
#inicial respectivamente para evitar problemas la posicion final se iguala a la posicion inicial antes
#de comenzar el programa

#Cuerpo 1
vic[0]=np.array([0., 0., 0.])
xic[0]=np.array([0., 0., 0.])
xfc[0]=xic[0]

#Cuerpo 2
vp=30.288e3-1000
pp=147.10e9+3.84e8
a=282.94 * ca
i=0.00 * ca
vic[1]=np.array([-vp*np.sin(a),vp*np.cos(a),0])
xic[1]=np.array([pp*np.cos(a)*np.cos(i),pp*np.sin(a)*np.cos(i),pp*np.sin(i)])
xfc[1]=xic[1]

#Cuerpo 3
vp=30.288e3
pp=147.10e9
a = 282.94 * ca
i = 0.00 * ca
vic[2]=np.array([-vp*np.sin(a),vp*np.cos(a),0])
xic[2]=np.array([pp*np.cos(a)*np.cos(i),pp*np.sin(a)*np.cos(i),pp*np.sin(i)])
xfc[2]=xic[2]

#Bucle para calcular la posicion de los cuerpos

#vectores de posicion de cada cuerpo
posiciones_cuerpo1 = []
posiciones_cuerpo2 = []
posiciones_cuerpo3 = []


n=0
while n<10000:
                            #SECCION 2-4-1
                            #El bucle se ejecutara siempre
                            #Calculo valores finales de posicion y velocidad de cada cuerpo
    n=n+1                   #Contador usado como opcion auxiliar para parar el programa
    tf=ti+dt
                            
    xfc,vfc=Runge_Kutta4(xic, vic, tf, ti, mc, numerocuerpos)

    # Guardar las posiciones de cada cuerpo en las listas
    posiciones_cuerpo1.append(xfc[0])
    posiciones_cuerpo2.append(xfc[1])
    posiciones_cuerpo3.append(xfc[2])

                            #Llama la funcion Ruge-Kutta para hallar los valores
                            #finales de posicion y velociadad
                            #Se actualizan los valores iniciales de velocidad, posicion y tiempo
    vic=vfc
    xic=xfc
    ti=tf

# Convertir listas a arrays para graficar
posiciones_cuerpo1 = np.array(posiciones_cuerpo1)
posiciones_cuerpo2 = np.array(posiciones_cuerpo2)
posiciones_cuerpo3 = np.array(posiciones_cuerpo3)

# Crear la gráfica
plt.figure()
plt.plot(posiciones_cuerpo1[:, 0], posiciones_cuerpo1[:, 1], label='Cuerpo 1')
plt.plot(posiciones_cuerpo2[:, 0], posiciones_cuerpo2[:, 1], label='Cuerpo 2')
plt.plot(posiciones_cuerpo3[:, 0], posiciones_cuerpo3[:, 1], label='Cuerpo 3')
plt.title('Trayectoria de los cuerpos')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.legend()
plt.grid(True)
plt.show()



