import numpy as np
import math
from vpython import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#defino derivada de la fuerza para calcular la velocidad de los cuerpos
def f_Velocidad(m1, m2, m3, pos1, pos2, pos3):
    G = 6.67e-11
    # Distancias totales entre los cuerpos
    r12 = np.linalg.norm(pos2 - pos1)
    r13 = np.linalg.norm(pos3 - pos1)
    r23 = np.linalg.norm(pos3 - pos2)
    
    # Vectores de distancia en cada dirección
    r12_vec = pos2 - pos1
    r13_vec = pos3 - pos1
    r23_vec = pos3 - pos2
    
    # Aceleraciones resultantes de las fuerzas gravitacionales en cada cuerpo
    V1 = -G * ((m2 * r12_vec / r12**3) + (m3 * r13_vec / r13**3))
    V2 = -G * ((m1 * -r12_vec / r12**3) + (m3 * r23_vec / r23**3))
    V3 = -G * ((m2 * -r23_vec / r23**3) + (m1 * -r13_vec / r13**3))
    
    return V1, V2, V3

#defino el cambio de posicion dependiendo de la velocidad y el dt
def f_cambioPosicion(pos1, pos2, pos3, vo1, vf1, vo2, vf2, vo3, vf3, dt):
    #X=Xo*t+0.5*a*t**2

    Xf1 = pos1[0]*dt+0.5*(vf1[0]-vo1[0])*dt**2
    Yf1 = pos1[1]*dt+0.5*(vf1[1]-vo1[1])*dt**2
    Xf2 = pos2[0]*dt+0.5*(vf2[0]-vo2[0])*dt**2
    Yf2 = pos2[1]*dt+0.5*(vf2[1]-vo2[1])*dt**2
    Xf3 = pos3[0]*dt+0.5*(vf3[0]-vo3[0])*dt**2
    Yf3 = pos3[1]*dt+0.5*(vf3[1]-vo3[1])*dt**2

    pos_F1 = np.array([Xf1, Yf1])
    pos_F2 = np.array([Xf2, Yf2])
    pos_F3 = np.array([Xf3, Yf3])

    return pos_F1, pos_F2, pos_F3

#Defino calculo de energia cinetica para un cuerpo
def e_Cinetica(mi, vi):
    return 0.5*mi*(np.linalg.norm(vi))**2

#defino calculo de energia potencial gravitacional entre dos cuerpos
def e_PotGravitacional(m1, m2, pos1, pos2):
    G = 6.67e-11
    return (-(G*m1*m2)/(np.linalg.norm(pos2-pos1)))

#defino calculo para la energia total del sistema
def e_Total(m1, m2, m3, v1, v2, v3, pos1, pos2, pos3):
    return (e_Cinetica(m1, v1) + e_Cinetica(m2, v2) + e_Cinetica(m3, v3))+ (e_PotGravitacional(m1, m2, pos1, pos2) + e_PotGravitacional(m1, m3, pos1, pos3) + e_PotGravitacional(m2, m3, pos2, pos3))

#defino calculo para la energia acumulativa
def e_Acumulativa(m1, m2, m3, velocidades1, velocidades2, velocidades3, posiciones1, posiciones2, posiciones3):   
    G = 6.67e-11
    eCinetica1 = 0,5*m1*((0.33*velocidades1[49]**3)-(0.33*velocidades1[0]**3))
    eCinetica2 = 0,5*m1*((0.33*velocidades2[49]**3)-(0.33*velocidades2[0]**3))
    eCinetica3 = 0,5*m1*((0.33*velocidades3[49]**3)-(0.33*velocidades3[0]**3))

    ePotGravitacional12 = -G*m1*m2*((np.linalg.norm(posiciones2[49]-posiciones1[49]))**(-2)-(np.linalg.norm(posiciones2[0]-posiciones1[0]))**(-2))
    ePotGravitacional13 = -G*m1*m3*((np.linalg.norm(posiciones3[49]-posiciones1[49]))**(-2)-(np.linalg.norm(posiciones3[0]-posiciones1[0]))**(-2))
    ePotGravitacional23 = -G*m2*m3*((np.linalg.norm(posiciones3[49]-posiciones2[49]))**(-2)-(np.linalg.norm(posiciones3[0]-posiciones2[0]))**(-2))

    return (eCinetica1+eCinetica2+eCinetica3+ePotGravitacional12+ePotGravitacional13+ePotGravitacional23)


#CONDICIONES INICIALES
#masas de los cuerpos
#m1 = 1
#m2 = 0.4
#m3 = 0.5

#posiciones iniciales
#posCuerpo1 = np.array([0., 0.])
#posCuerpo2 = np.array([15., 12.])
#posCuerpo3 = np.array([5.4, 4.5])

#velocidades iniciales
#v_inicial1 = np.array([0.0, 0.0]) 
#v_inicial2 = np.array([1.2, -1.2])
#v_inicial3 = np.array([-1.5, 1.3])

#CONDICIONES INICIALES: ORBITA TRIANGULAR DE LAGRANGE
# Posiciones iniciales en triángulo equilátero
#posCuerpo1 = np.array([0., 0.])
#posCuerpo2 = np.array([1., np.sqrt(3)])
#posCuerpo3 = np.array([-1., np.sqrt(3)])

# Velocidades iniciales (ajustar según sea necesario para la estabilidad)
#v_inicial1 = np.array([0.5, -0.5])
#v_inicial2 = np.array([-0.5, -0.5])
#v_inicial3 = np.array([0.0, 1.0])

#CONDICIONES INICIALES: ORBITAS EN FORMA DE 8 DE CHENCINER Y MONTGOMERY
# Asume masas iguales para los cuerpos
m1 = m2 = m3 = 1.0

# Posiciones iniciales aproximadas
#posCuerpo1 = np.array([0.970, 0.243])
#posCuerpo2 = np.array([-0.970, 0.243])
#posCuerpo3 = np.array([0.0, -0.486])

# Velocidades iniciales aproximadas
#v_inicial1 = np.array([0.466, 0.432])
#v_inicial2 = np.array([0.466, 0.432])
#v_inicial3 = np.array([-0.932, -0.864])

#CONDICIONES INICIALES: ORBITA COLINEAL DE EULER
#Asumimos las masas iguales de la condicion anterior
#Posiciones iniciales aproximadas
posCuerpo1 = np.array([1.0, 0.0])
posCuerpo2 = np.array([6.0, 0.0])
posCuerpo3 = np.array([3.0, 0.0])

#Velocidades iniciales aproximadas
v_inicial1 = np.array([0.466, 0.432])
v_inicial2 = np.array([0.466, 0.432])
v_inicial3 = np.array([-0.932, -0.864])


#inicio vector de trayectoria:
trayectoria_cuerpo1 = [posCuerpo1] 
trayectoria_cuerpo2 = [posCuerpo2]
trayectoria_cuerpo3 = [posCuerpo3]

#Energia inicial
#cinetica
print("Energia cinetica cuerpo 1: ")
print(e_Cinetica(m1, v_inicial1))
print("Energia cinetica cuerpo 2: ")
print(e_Cinetica(m2, v_inicial2))
print("Energia cinetica cuerpo 3: ")
print(e_Cinetica(m3, v_inicial3))

#gravitacional
print("Energia gravitacional entre cuerpo 1 y 2: ")
print(e_PotGravitacional(m1,m2, posCuerpo1, posCuerpo2))
print("Energia gravitacional entre cuerpo 1 y 3: ")
print(e_PotGravitacional(m1,m3, posCuerpo1, posCuerpo3))
print("Energia gravitacional entre cuerpo 2 y 3: ")
print(e_PotGravitacional(m2,m3, posCuerpo2, posCuerpo3))

#Total
print("Energia total del sistema: ")
print(e_Total(m1, m2, m3, v_inicial1, v_inicial2, v_inicial3, posCuerpo1, posCuerpo2, posCuerpo3))

#arreglos con las velocidades de los cuerpos
velocidadesCuerpo1 = [v_inicial1]
velocidadesCuerpo2 = [v_inicial2]
velocidadesCuerpo3 = [v_inicial3]

#listas para almacenar las energías
energias_cineticas = []
energias_potenciales = []
energias_totales = []

t = 0
i=0
dt = 0.1

while t < 50:
    t += dt

    


""" 
trayectoria_cuerpo1 = np.array(trayectoria_cuerpo1)
trayectoria_cuerpo2 = np.array(trayectoria_cuerpo2)
trayectoria_cuerpo3 = np.array(trayectoria_cuerpo3)

# Configuración de la animación
fig, ax = plt.subplots()
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel("Posición X (m)")
ax.set_ylabel("Posición Y (m)")
ax.set_title("Trayectorias de los cuerpos en el sistema")

# Puntos de los cuerpos
punto1, = ax.plot([], [], 'bo', label='Cuerpo 1')
punto2, = ax.plot([], [], 'ro', label='Cuerpo 2')
punto3, = ax.plot([], [], 'go', label='Cuerpo 3')
# Líneas de trayectoria
trayectoria1, = ax.plot([], [], 'b--')
trayectoria2, = ax.plot([], [], 'r--')
trayectoria3, = ax.plot([], [], 'g--')
ax.legend()

# Función de actualización para la animación
def actualizar(frame):
    punto1.set_data([trayectoria_cuerpo1[frame, 0]], [trayectoria_cuerpo1[frame, 1]])
    punto2.set_data([trayectoria_cuerpo2[frame, 0]], [trayectoria_cuerpo2[frame, 1]])
    punto3.set_data([trayectoria_cuerpo3[frame, 0]], [trayectoria_cuerpo3[frame, 1]])

    trayectoria1.set_data(trayectoria_cuerpo1[:frame, 0], trayectoria_cuerpo1[:frame, 1])
    trayectoria2.set_data(trayectoria_cuerpo2[:frame, 0], trayectoria_cuerpo2[:frame, 1])
    trayectoria3.set_data(trayectoria_cuerpo3[:frame, 0], trayectoria_cuerpo3[:frame, 1])
    return punto1, punto2, punto3, trayectoria1, trayectoria2, trayectoria3

# Crear la animación
anim = FuncAnimation(fig, actualizar, frames=len(trayectoria_cuerpo1), interval=50, blit=True)

plt.show()

"""
