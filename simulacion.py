import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go

#defino la funcion aceleracion Fgrav/mi
def f_aceleracion(m1, m2, m3, pos1, pos2, pos3):
    G = 6.67e-11
    posiciones = np.array([pos1, pos2, pos3])
    masas = np.array([m1, m2, m3])
    aceleraciones = np.zeros_like(posiciones, dtype=np.float64)
    cuerpos = 3
    for i in range(cuerpos):
        for j in range(cuerpos):
            if j!=i:
                r = posiciones[j]-posiciones[i]
                dist = np.linalg.norm(r)
                if dist>0:
                    aceleraciones[i] += G*masas[j]*r/(dist**3)
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
def f_posicion(m1, m2, m3, posicionT1, posicionT2, posicionT3, h, velocidadT1, velocidadT2, velocidadT3):
    v1, v2, v3 = f_Velocidad(m1, m2, m3, velocidadT1, velocidadT2, velocidadT3, h, posicionT1, posicionT2, posicionT3)
    pos1 = posicionT1 + v1*h 
    pos2 = posicionT2 + v2*h 
    pos3 = posicionT3 + v3*h 
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
    # Calcular la distancia entre los cuerpos en función de sus posiciones
    dist_12 = np.linalg.norm(posC1 - posC2)  # Distancia entre Cuerpo 1 y Cuerpo 2
    dist_13 = np.linalg.norm(posC1 - posC3)  # Distancia entre Cuerpo 1 y Cuerpo 3
    dist_23 = np.linalg.norm(posC2 - posC3)  # Distancia entre Cuerpo 2 y Cuerpo 3
    
    # Cálculo de la energía potencial entre los cuerpos
    potGrav1 = -G * m1 * (m2 / dist_12 + m3 / dist_13)
    potGrav2 = -G * m2 * (m1 / dist_12 + m3 / dist_23)
    potGrav3 = -G * m3 * (m1 / dist_13 + m2 / dist_23)
    return potGrav1, potGrav2, potGrav3

#defino la funcion de energia acumulativa
def energiaAcumulativa(eTotal0, eTotalT):
    return (eTotalT-eTotal0)

def trapecio(integrand, dt):
    integral = 0.0
    for i in range(1, len(integrand)):
        integral += (integrand[i-1] + integrand[i]) / 2 * dt
    return integral

def newton_cotes(integrand, dt):
    n = len(integrand)
    if n < 3:
        raise ValueError("Se requieren al menos 3 puntos para aplicar Newton-Cotes.")
    
    integral = 0.0
    for i in range(0, n-2, 2):
        integral += (integrand[i] + 4*integrand[i+1] + integrand[i+2]) * (dt/3)

    # Si hay un punto adicional, sumarlo
    if n % 2 == 0:
        integral += (integrand[-2] + integrand[-1]) * (dt/2)

    return integral

def gauss_quadrature(integrand, dt):
    # Puntos y pesos para 2 puntos de Gauss
    puntos = [1/2 - np.sqrt(3)/6, 1/2 + np.sqrt(3)/6]
    pesos = [1/2, 1/2]
    
    integral = 0.0
    for i in range(len(integrand) - 1):
        for j in range(2):
            # Evaluar en los puntos de Gauss
            x = (integrand[i] + integrand[i + 1]) / 2 + (puntos[j] * (integrand[i + 1] - integrand[i])) / 2
            integral += pesos[j] * x * dt

    return integral

#SISTEMA TIERRA, MARTE, SOL. EL QUE MAS ME GUSTA
# Masas (en kg)
m1 = 5.972e24   # Masa de la Tierra
m2 = 6.4171e23  # Masa de Marte
m3 = 1.989e30   # Masa del Sol

# Posiciones iniciales (en metros)
posCuerpo1 = np.array([1.496e11, 0, 0])               # Posición inicial de la Tierra
posCuerpo2 = np.array([2.279e11, 0, 0])               # Posición inicial de Marte
posCuerpo3 = np.array([0, 0, 0])                      # Posición inicial del Sol (en el origen)

# Velocidades iniciales (en m/s)
v_inicial1 = np.array([0, 29780, 0])                  # Velocidad inicial de la Tierra
v_inicial2 = np.array([0, 24100, 0])                  # Velocidad inicial de Marte
v_inicial3 = np.array([0, 0, 0])                      # Velocidad inicial del Sol (suponemos que está en reposo)

# Inicializar variables para almacenar posiciones, velocidades y energías
trayectoriaCuerpo1 = []
trayectoriaCuerpo2 = []
trayectoriaCuerpo3 = []

velocidades_cuerpo1 = []
velocidades_cuerpo2 = []
velocidades_cuerpo3 = []

eCineticaCuerpo1 = []
eCineticaCuerpo2 = []
eCineticaCuerpo3 = []

ePotGravitacional1 = []
ePotGravitacional2 = []
ePotGravitacional3 = []

eTotal = []

# Bucle principal para la simulación
while t < t_max:
    # Calcular la aceleración (fuerza gravitatoria)
    aceleraciones = f_aceleracion(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3)
    
    # Calcular la nueva velocidad con la aceleración calculada
    v_inicial1, v_inicial2, v_inicial3 = f_Velocidad(m1, m2, m3, v_inicial1, v_inicial2, v_inicial3, dt, posCuerpo1, posCuerpo2, posCuerpo3)
    
    # Calcular la nueva posición con la velocidad calculada
    posCuerpo1, posCuerpo2, posCuerpo3 = f_posicion(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3, dt, v_inicial1, v_inicial2, v_inicial3)
    
    # Almacenar las posiciones y velocidades
    trayectoriaCuerpo1.append(posCuerpo1)
    trayectoriaCuerpo2.append(posCuerpo2)
    trayectoriaCuerpo3.append(posCuerpo3)
    
    velocidades_cuerpo1.append(v_inicial1)
    velocidades_cuerpo2.append(v_inicial2)
    velocidades_cuerpo3.append(v_inicial3)
    
    # Calcular y almacenar las energías cinéticas y potenciales
    e_cinetica1, e_cinetica2, e_cinetica3 = energiaCinetica(m1, m2, m3, v_inicial1, v_inicial2, v_inicial3)
    ePot1, ePot2, ePot3 = energiaPotencialGrav(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3)
    
    eCineticaCuerpo1.append(e_cinetica1)
    eCineticaCuerpo2.append(e_cinetica2)
    eCineticaCuerpo3.append(e_cinetica3)
    
    ePotGravitacional1.append(ePot1)
    ePotGravitacional2.append(ePot2)
    ePotGravitacional3.append(ePot3)
    
    # Energía total
    eTotal.append(e_cinetica1 + e_cinetica2 + e_cinetica3 + ePot1 + ePot2 + ePot3)
    
    # Incrementar el tiempo
    t += dt



# [Las funciones anteriores permanecen igual hasta el bucle de simulación]

# Después de recopilar los datos, modificamos la visualización:

# Convertir las listas de trayectorias a arrays de numpy
trayectoriaCuerpo1 = np.array(trayectoriaCuerpo1)
trayectoriaCuerpo2 = np.array(trayectoriaCuerpo2)
trayectoriaCuerpo3 = np.array(trayectoriaCuerpo3)

# Crear figura para Matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.cla()  # Limpiar el eje en cada frame
    
    actual_frame = frame * 10000  # Para que vaya más rápido
    
    # Dibujar las trayectorias
    ax.plot(trayectoriaCuerpo1[:actual_frame, 0], trayectoriaCuerpo1[:actual_frame, 1], trayectoriaCuerpo1[:actual_frame, 2], 'b-', alpha=0.5, label='Cuerpo 1')
    ax.plot(trayectoriaCuerpo2[:actual_frame, 0], trayectoriaCuerpo2[:actual_frame, 1], trayectoriaCuerpo2[:actual_frame, 2], 'r-', alpha=0.5, label='Cuerpo 2')
    ax.plot(trayectoriaCuerpo3[:actual_frame, 0], trayectoriaCuerpo3[:actual_frame, 1], trayectoriaCuerpo3[:actual_frame, 2], 'g-', alpha=0.5, label='Cuerpo 3')
    
    # Dibujar los cuerpos
    ax.scatter(trayectoriaCuerpo1[actual_frame, 0], trayectoriaCuerpo1[actual_frame, 1], trayectoriaCuerpo1[actual_frame, 2], color='blue', s=100)
    ax.scatter(trayectoriaCuerpo2[actual_frame, 0], trayectoriaCuerpo2[actual_frame, 1], trayectoriaCuerpo2[actual_frame, 2], color='red', s=100)
    ax.scatter(trayectoriaCuerpo3[actual_frame, 0], trayectoriaCuerpo3[actual_frame, 1], trayectoriaCuerpo3[actual_frame, 2], color='green', s=100)
    
    # Configurar límites y etiquetas
    all_positions = np.vstack([trayectoriaCuerpo1, trayectoriaCuerpo2, trayectoriaCuerpo3])
    margin = 0.1 * (np.max(all_positions) - np.min(all_positions))
    ax.set_xlim(np.min(all_positions[:, 0]) - margin, np.max(all_positions[:, 0]) + margin)
    ax.set_ylim(np.min(all_positions[:, 1]) - margin, np.max(all_positions[:, 1]) + margin)
    ax.set_zlim(np.min(all_positions[:, 2]) - margin, np.max(all_positions[:, 2]) + margin)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Simulación de Tres Cuerpos - Frame {actual_frame}')
    ax.legend()

# Crear la animación
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=len(trayectoriaCuerpo1)//10,
    interval=10,
    repeat=True
)

plt.show()

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)

print("Energia acumulativa del sistema: ",  sum(eTotal) * dt)
print("Energia Acumulada con la funcion: ", energiaAcumulativa(eTotal[0], eTotal[-1]))
print(f"Energía acumulada (Trapecio): {energia_acumulada_trapecio:.2f}")
print(f"Energía acumulada (Newton-Cotes): {energia_acumulada_newton_cotes:.2f}")
print(f"Energía acumulada (Cuadratura de Gauss): {energia_acumulada_gauss:.2f}")