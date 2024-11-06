import numpy as np
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

#defino la funcion aceleracion Fgrav/mi
def f_aceleracion(m1, m2, m3, pos1, pos2, pos3):
    G = 6.67e-11
    aceleraciones = np.array([[0,0], [0,0], [0,0]], dtype=np.float64)
    cuerpos=3
    posiciones = np.array([pos1, pos2, pos3])
    masas = np.array([m1, m2, m3])
    for i in range(cuerpos):
        for j in range(cuerpos):
            if j!=i:
                r = posiciones[j]-posiciones[i]
                dist = np.linalg.norm(r)
                if dist > 0:
                    aceleraciones[i] += G * masas[j] * r/(dist**3)
    return aceleraciones



#defino la funcion velocidad
""" deberia hacer la integral de t a t+h de la funcion gravitatoria, 
lo unico que cambiara es la posicion en funcion del tiempo"""

def f_Velocidad(m1, m2, m3, vC1, vC2, vC3, h, pos1, pos2, pos3):
    aceleraciones = f_aceleracion(m1, m2, m3, pos1, pos2, pos3)
    v_1 = vC1 + aceleraciones[0]*h
    v_2 = vC2 + aceleraciones[1]*h
    v_3 = vC3 + aceleraciones[2]*h
    return v_1, v_2, v_3

#defino la funcion de posicion 
""" al igual que con la velocidad, debo hacer la integral de la funcin de t a t+h
en la funcion de velocidad"""
def f_posicion(m1, m2, m3, posicionT1, posicionT2, posicionT3, h, velocidadT1, velocidadT2, velocidadT3, pos1, pos2, pos3):
    v1, v2, v3 = f_Velocidad(m1, m2, m3, velocidadT1, velocidadT2, velocidadT3, h, pos1, pos2, pos3)
    pos1 = posicionT1 + v1 * h  
    pos2 = posicionT2 + v2 * h 
    pos3 = posicionT3 + v3 * h 
    return pos1, pos2, pos3

#defino la funcion de energia cinetica para cada cuerpo
def energiaCinetica(m1, m2, m3, vC1, vC2, vC3):
    cinetica1 = 0.5*m1*(np.linalg.norm(vC1)**2)
    cinetica2 = 0.5*m2*(np.linalg.norm(vC2)**2)
    cinetica3 = 0.5*m3*(np.linalg.norm(vC3)**2)
    return cinetica1, cinetica2, cinetica3

#defino la funcion para la energia potencial entre dos cuerpos
def energiaPotencialGrav(m1, m2, m3, posC1, posC2, posC3):
    G = 6.67e-11
    E12 = -G * m1 * m2 / np.linalg.norm(posC2-posC1)
    E13 = -G * m1 * m3 / np.linalg.norm(posC3-posC1)
    E23 = -G * m2 * m3 / np.linalg.norm(posC3-posC2)
    
    # Distribuir la energía potencial entre los cuerpos
    potGrav1 = E12 + E13
    potGrav2 = E12 + E23
    potGrav3 = E13 + E23
    
    return potGrav1, potGrav2, potGrav3

#defino la funcion de energia acumulativa
def energiaAcumulativa(eTotal0, eTotalT):
    return (eTotalT-eTotal0)

#CONDICIONES INICIALES
m1 = 5.97e24 #tierra
m2 = 7.348e22 #luna
m3 = 1.98e30 #sol

#Posiciones iniciales aproximadas
posCuerpo1 = np.array([1.4961e11, 0.0])
posCuerpo2 = np.array([7.4805e10, 1.2956e11])
posCuerpo3 = np.array([0.0, 0.0])

#Velocidades iniciales aproximadas
v_inicial1 = np.array([0.0, 29780.0])
v_inicial2 = np.array([0.0, 30802.0])
v_inicial3 = np.array([0.0, 0.0])

#inicializar trayectorias
trayectoriaCuerpo1 = []
trayectoriaCuerpo2 = []
trayectoriaCuerpo3 = []

#inicializar velocidades
velocidades_cuerpo1 = []
velocidades_cuerpo2 = []
velocidades_cuerpo3 = []

#arrglos con energias cineticas y potenciales para cada cuerpo
eCineticaCuerpo1 = []
eCineticaCuerpo2 = []
eCineticaCuerpo3 = []
ePotGravitacional1 = []
ePotGravitacional2 = []
ePotGravitacional3 = []
eTotal = []
t=0
dt=60 #segundos en un minuto
t_max = 31536000 #segundos en un anio
fuerzaGravitatoriaC1, fuerzaGravitatoriaC2, fuerzaGravitatoriaC3 = fuerzaGravitatoria(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3)

aceleracion = f_aceleracion(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3)
print("ACELERACIONES GRAVITATORIAS")
print("ACELERACION CUERPO 1", aceleracion[0])
print("ACELERACION CUERPO 2", aceleracion[1])
print("ACELERACION CUERPO 3", aceleracion[2])


while t<t_max:
    #calculo de la velocidad en funcion de la fuerza grav
    v_cuerpo1, v_cuerpo2, v_cuerpo3 = f_Velocidad(m1, m2, m3, v_inicial1, v_inicial2, v_inicial3, dt, posCuerpo1, posCuerpo2, posCuerpo3)

    #calculo de la posicion en funcion de la velocidad
    posNueva_cuerpo1, posNueva_cuerpo2, posNueva_cuerpo3 = f_posicion(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3, dt,
                                                              v_cuerpo1, v_cuerpo2, v_cuerpo3, posCuerpo1, posCuerpo2, posCuerpo3)
     
    #actualizo las variables
    posCuerpo1 = posNueva_cuerpo1
    posCuerpo2 = posNueva_cuerpo2
    posCuerpo3 = posNueva_cuerpo3
    v_inicial1 = v_cuerpo1
    v_inicial2 = v_cuerpo2
    v_inicial3 = v_cuerpo3

    #añado los nuevos calculos a los respectivos arreglos
    velocidades_cuerpo1.append(v_inicial1)
    velocidades_cuerpo2.append(v_inicial2)
    velocidades_cuerpo3.append(v_inicial3)

    trayectoriaCuerpo1.append(posCuerpo1)
    trayectoriaCuerpo2.append(posCuerpo2)
    trayectoriaCuerpo3.append(posCuerpo3)

    #calculo y actualizacion de energias
    e_cinetica1, e_cinetica2, e_cinetica3 = energiaCinetica(m1, m2, m3, v_inicial1, v_inicial2, v_inicial3)
    eCineticaCuerpo1.append(e_cinetica1)
    eCineticaCuerpo2.append(e_cinetica2)
    eCineticaCuerpo3.append(e_cinetica3)

    e_potencial1, e_potencial2, e_potencial3 = energiaPotencialGrav(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3)
    ePotGravitacional1.append(e_potencial1)
    ePotGravitacional2.append(e_potencial2)
    ePotGravitacional3.append(e_potencial3)

    e_cineticaTotal = e_cinetica1+e_cinetica2+e_cinetica3
    e_potGravTotal = e_potencial1+e_potencial2+e_potencial3

    eTotal.append((e_cineticaTotal+e_potGravTotal))
    
    t += dt

print(f"Energía Cinética - Cuerpo 1: {eCineticaCuerpo1[-1]:.2f}")
print(f"Energía Cinética - Cuerpo 2: {eCineticaCuerpo2[-1]:.2f}")
print(f"Energía Cinética - Cuerpo 3: {eCineticaCuerpo3[-1]:.2f}")
print(f"Energía Potencial - Cuerpo 1: {ePotGravitacional1[-1]:.2f}")
print(f"Energía Potencial - Cuerpo 2: {ePotGravitacional2[-1]:.2f}")
print(f"Energía Potencial - Cuerpo 3: {ePotGravitacional3[-1]:.2f}")
print(f"Energía Total del Sistema: {eTotal[-1]:.2f}")
Ultimaposicion = len(eTotal)-1
print(f"Energia Acumulativa del Sistema:", energiaAcumulativa(eTotal[0], eTotal[Ultimaposicion]))

# Configuración de la animación
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)
ax.set_aspect('equal')
ax.grid(True)

# Elementos de la gráfica
linea_trayectoria1, = ax.plot([], [], 'b-', label='Cuerpo 1', alpha=0.5)
linea_trayectoria2, = ax.plot([], [], 'r-', label='Cuerpo 2', alpha=0.5)
linea_trayectoria3, = ax.plot([], [], 'g-', label='Cuerpo 3', alpha=0.5)
punto_cuerpo1, = ax.plot([], [], 'bo', markersize=10)
punto_cuerpo2, = ax.plot([], [], 'ro', markersize=10)
punto_cuerpo3, = ax.plot([], [], 'go', markersize=10)

# Título y leyenda
ax.set_title('Simulación del Problema de los Tres Cuerpos')
ax.legend()

def init():
    #Función de inicialización para la animación
    linea_trayectoria1.set_data([], [])
    linea_trayectoria2.set_data([], [])
    linea_trayectoria3.set_data([], [])
    punto_cuerpo1.set_data([], [])
    punto_cuerpo2.set_data([], [])
    punto_cuerpo3.set_data([], [])
    return linea_trayectoria1, linea_trayectoria2, linea_trayectoria3, punto_cuerpo1, punto_cuerpo2, punto_cuerpo3

def update(frame):
    #Función de actualización para cada frame de la animación
    global posCuerpo1, posCuerpo2, posCuerpo3, v_inicial1, v_inicial2, v_inicial3
    
    # Calcular nuevas posiciones
    pos_nueva1, pos_nueva2, pos_nueva3 = f_posicion(
        m1, m2, m3,
        posCuerpo1, posCuerpo2, posCuerpo3,
        0.1,  # dt
        v_inicial1, v_inicial2, v_inicial3,
        posCuerpo1, posCuerpo2, posCuerpo3
    )
    
    # Calcular nuevas velocidades
    v_nueva1, v_nueva2, v_nueva3 = f_Velocidad(
        m1, m2, m3,
        v_inicial1, v_inicial2, v_inicial3,
        0.1,  # dt
        posCuerpo1, posCuerpo2, posCuerpo3
    )
    
    # Actualizar posiciones y velocidades
    posCuerpo1, posCuerpo2, posCuerpo3 = pos_nueva1, pos_nueva2, pos_nueva3
    v_inicial1, v_inicial2, v_inicial3 = v_nueva1, v_nueva2, v_nueva3
    
    # Agregar nuevas posiciones a las trayectorias
    trayectoriaCuerpo1.append(posCuerpo1)
    trayectoriaCuerpo2.append(posCuerpo2)
    trayectoriaCuerpo3.append(posCuerpo3)
    
    # Actualizar datos de las trayectorias
    traj1 = np.array(trayectoriaCuerpo1)
    traj2 = np.array(trayectoriaCuerpo2)
    traj3 = np.array(trayectoriaCuerpo3)
    
    linea_trayectoria1.set_data(traj1[:, 0], traj1[:, 1])
    linea_trayectoria2.set_data(traj2[:, 0], traj2[:, 1])
    linea_trayectoria3.set_data(traj3[:, 0], traj3[:, 1])
    
    # Actualizar posiciones actuales de los cuerpos
    punto_cuerpo1.set_data([posCuerpo1[0]], [posCuerpo1[1]])
    punto_cuerpo2.set_data([posCuerpo2[0]], [posCuerpo2[1]])
    punto_cuerpo3.set_data([posCuerpo3[0]], [posCuerpo3[1]])
    
    return linea_trayectoria1, linea_trayectoria2, linea_trayectoria3, punto_cuerpo1, punto_cuerpo2, punto_cuerpo3

# Crear la animación
anim = FuncAnimation(fig, update, init_func=init, frames=500, interval=20, blit=True)
plt.show()
