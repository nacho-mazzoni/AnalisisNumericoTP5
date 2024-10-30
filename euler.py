import numpy as np
import math
from vpython import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#defino la funcion de fuerza gravitatoria
def fuerzaGravitatoria(m1, m2 ,m3, pos1, pos2, pos3):
    G = 6.67e-11
    f_Grav1 = -G*m1*((m2*(pos2-pos1)/(np.linalg.norm(pos2-pos1)**3))+(m3*(pos3-pos1)/(np.linalg.norm(pos3-pos1)**3)))
    f_Grav2 = -G*m2*((m1*(pos1-pos2)/(np.linalg.norm(pos1-pos2)**3))+(m3*(pos3-pos2)/(np.linalg.norm(pos3-pos2)**3)))
    f_Grav3 = -G*m3*((m1*(pos1-pos3)/(np.linalg.norm(pos1-pos3)**3))+(m2*(pos2-pos3)/(np.linalg.norm(pos2-pos3)**3)))

    return f_Grav1, f_Grav2, f_Grav3

""" se reciben las masas y la ultima posicion actualizada hasta ese momento t """

#defino la funcion velocidad
""" deberia hacer la integral de t a t+h de la funcion gravitatoria, 
lo unico que cambiara es la posicion en funcion del tiempo"""

def f_Velocidad(m1, m2, m3, vC1, vC2, vC3, h, pos1, pos2, pos3):
    f_grav1, f_grav2, f_grav3 = fuerzaGravitatoria(m1, m2, m3, pos1, pos2, pos3)

    #print("FUNCION VELOCIDAD")

    # Imprime los valores de los argumentos que recibe la función
    #print(f"Argumentos recibidos: m1={m1}, m2={m2}, m3={m3}")
    #print(f"Posiciones: pos1={pos1}, pos2={pos2}, pos3={pos3}")
    #print(f"Velocidades iniciales: velocidadT1={vC1}, velocidadT2={vC2}, velocidadT3={vC3}")
    #print(f"Fuerzas gravitatorias: f_grav1={f_grav1}, f_grav2={f_grav2}, f_grav3={f_grav3}")

    v_1 = vC1 + (h/m1)*f_grav1
    v_2 = vC2 + (h/m2)*f_grav2
    v_3 = vC3 + (h/m3)*f_grav3

    # Imprime las velocidades calculadas antes de retornarlas
    #print(f"Nuevas velocidades: v_1={v_1}, v_2={v_2}, v_3={v_3}")
    #print("-" * 40)  # Separador para mayor claridad en la salida

    return v_1, v_2, v_3

#defino la funcion de posicion 
""" al igual que con la velocidad, debo hacer la integral de la funcin de t a t+h
en la funcion de velocidad"""
def f_posicion(m1, m2, m3, posicionT1, posicionT2, posicionT3, dt, velocidadT1, velocidadT2, velocidadT3, pos1, pos2, pos3):
    v1, v2, v3 = f_Velocidad(m1, m2, m3, velocidadT1, velocidadT2, velocidadT3, dt, pos1, pos2, pos3)
    pos1 = posicionT1 + dt*v1
    pos2 = posicionT2 + dt*v2
    pos3 = posicionT3 + dt*v3
    return pos1, pos2, pos3


#CONDICIONES INICIALES: ORBITAS EN FORMA DE 8 DE CHENCINER Y MONTGOMERY
# Asume masas iguales para los cuerpos
m1 = 1
m2 = 1
m3 = 1

#Posiciones iniciales aproximadas
posCuerpo1 = np.array([0, -5])
posCuerpo2 = np.array([6.0, 6.0])
posCuerpo3 = np.array([-35.0, 0.0])

#Velocidades iniciales aproximadas
v_inicial1 = np.array([0.466, 0.432])
v_inicial2 = np.array([0.466, 0.432])
v_inicial3 = np.array([-0.932, -0.864])

#inicializar trayectorias
trayectoriaCuerpo1 = []
trayectoriaCuerpo2 = []
trayectoriaCuerpo3 = []

#inicializar velocidades
velocidades_cuerpo1 = []
velocidades_cuerpo2 = []
velocidades_cuerpo3 = []

t=0
dt=0.1

while t<50:
    #calculo de la velocidad en funcion de la fuerza grav
    v_cuerpo1, v_cuerpo2, v_cuerpo3 = f_Velocidad(m1, m2, m3, v_inicial1, v_inicial2, v_inicial3, dt, posCuerpo1, posCuerpo2, posCuerpo3)
    #print("imprime antes de acrualizar las velocidades")
    #print(f"Tiempo: {t:.2f}, Velocidades: {v_cuerpo1}, {v_cuerpo2}, {v_cuerpo3}")

    #calculo de la posicion en funcion de la velocidad
    posNueva_cuerpo1, posNueva_cuerpo2, posNueva_cuerpo3 = f_posicion(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3, dt,
                                                              v_cuerpo1, v_cuerpo2, v_cuerpo3, posCuerpo1, posCuerpo2, posCuerpo3)
    
    """if (t%10) == 0:
        #compruebo las dimensiones antes de actualizar
        print(f"Dimensiones después de f_velocidad - v_cuerpo1: {v_cuerpo1.shape}, v_cuerpo2: {v_cuerpo2.shape}, v_cuerpo3: {v_cuerpo3.shape}")
        print(f"Dimensiones después de f_posicion - posCuerpo1: {posNueva_cuerpo1.shape}, posCuerpo2: {posNueva_cuerpo2.shape}, posCuerpo3: {posNueva_cuerpo3.shape}")
    """    
    #actualizo las variables
    posCuerpo1 = posNueva_cuerpo1
    posCuerpo2 = posNueva_cuerpo2
    posCuerpo3 = posNueva_cuerpo3
    v_inicial1 = v_cuerpo1
    v_inicial2 = v_cuerpo2
    v_inicial3 = v_cuerpo3

    #print(f"Tiempo: {t}")
    #print(f"Posiciones iniciales: {posCuerpo1}, {posCuerpo2}, {posCuerpo3}")
    #print(f"Velocidades iniciales: {v_inicial1}, {v_inicial2}, {v_inicial3}")

    #añado los nuevos calculos a los respectivos arreglos
    velocidades_cuerpo1.append(v_inicial1)
    velocidades_cuerpo2.append(v_inicial2)
    velocidades_cuerpo3.append(v_inicial3)

    trayectoriaCuerpo1.append(posCuerpo1)
    trayectoriaCuerpo2.append(posCuerpo2)
    trayectoriaCuerpo3.append(posCuerpo3)

    #print(f"Dimensiones de posCuerpo1, posCuerpo2, posCuerpo3: {posCuerpo1.shape}, {posCuerpo2.shape}, {posCuerpo3.shape}")
    #print(f"Dimensiones de v_inicial1, v_inicial2, v_inicial3: {v_inicial1.shape}, {v_inicial2.shape}, {v_inicial3.shape}")

    t += dt


# Convertir las trayectorias a arrays de numpy para facilitar la animación
trayectoriaCuerpo1 = np.array(trayectoriaCuerpo1)
trayectoriaCuerpo2 = np.array(trayectoriaCuerpo2)
trayectoriaCuerpo3 = np.array(trayectoriaCuerpo3)

#Configuración de la gráfica
plt.figure(figsize=(10, 10))
plt.xlim(-10, 10)  # Ajusta estos límites según tus necesidades
plt.ylim(-10, 10)

# Graficar trayectorias
plt.plot(trayectoriaCuerpo1[:, 0], trayectoriaCuerpo1[:, 1], 'b-', label='Cuerpo 1')
plt.plot(trayectoriaCuerpo2[:, 0], trayectoriaCuerpo2[:, 1], 'r-', label='Cuerpo 2')
plt.plot(trayectoriaCuerpo3[:, 0], trayectoriaCuerpo3[:, 1], 'g-', label='Cuerpo 3')

# Graficar posiciones finales
plt.plot(trayectoriaCuerpo1[-1, 0], trayectoriaCuerpo1[-1, 1], 'bo')  # Cuerpo 1
plt.plot(trayectoriaCuerpo2[-1, 0], trayectoriaCuerpo2[-1, 1], 'ro')  # Cuerpo 2
plt.plot(trayectoriaCuerpo3[-1, 0], trayectoriaCuerpo3[-1, 1], 'go')  # Cuerpo 3

# Configuración de la gráfica
plt.title('Trayectorias de los Cuerpos')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.legend()
plt.grid()
plt.axis('equal')  # Para mantener la proporción de la gráfica
plt.show()