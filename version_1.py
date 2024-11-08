import numpy as np
import matplotlib.pyplot as plt

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
def f_posicion(m1, m2, m3, posicionT1, posicionT2, posicionT3, h, velocidadT1, velocidadT2, velocidadT3, pos1, pos2, pos3):
    v1, v2, v3 = f_Velocidad(m1, m2, m3, velocidadT1, velocidadT2, velocidadT3, h, pos1, pos2, pos3)
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

#CONDICIONES INICIALES

#SISTEMA TIERRA, LUNA, SOL
# Masas (en kg)
m1 = 5.972e24   # Masa de la Tierra
m2 = 7.348e22   # Masa de la Luna
m3 = 1.989e30   # Masa del Sol

# Posiciones iniciales (en metros)
posCuerpo1 = np.array([1.496e11, 0])               # Posición inicial de la Tierra
posCuerpo2 = np.array([1.496e11 + 3.844e8, 0])     # Posición inicial de la Luna
posCuerpo3 = np.array([0, 0])                      # Posición inicial del Sol

# Velocidades iniciales (en m/s)
v_inicial1 = np.array([0, 29780])                  # Velocidad inicial de la Tierra
v_inicial2 = np.array([0, 29780 + 1022])           # Velocidad inicial de la Luna
v_inicial3 = np.array([0, 0])                      # Velocidad inicial del Sol (en reposo)


#CON EL QUE EMPEZAMOS
# Masas (en kg)
m1 = 5.97e24
m2 = 7.348e22
m3 = 1.98e30  

# Posiciones iniciales (en metros)
posCuerpo1 = np.array([1.4961e11, 0])   # Tierra
posCuerpo2 = np.array([1.4961e11*np.cos(np.pi/3), 1.4961e11*np.sin(np.pi/3)])  # Luna
posCuerpo3 = np.array([0, 0])                     # Sol

# Velocidades iniciales (en m/s)
v_inicial1 = np.array([0, 29780])
v_inicial2 = np.array([0, -30802])
v_inicial3 = np.array([0, 0])

#SISTEMA SOL, TIERRA, MARTE
# Masas (en kg)
m1 = 5.972e24   # Masa de la Tierra
m2 = 6.4171e23  # Masa de Marte
m3 = 1.989e30   # Masa del Sol

# Posiciones iniciales (en metros)
posCuerpo1 = np.array([1.496e11, 0])               # Posición inicial de la Tierra
posCuerpo2 = np.array([2.279e11, 0])               # Posición inicial de Marte
posCuerpo3 = np.array([0, 0])                      # Posición inicial del Sol (en el origen)

# Velocidades iniciales (en m/s)
v_inicial1 = np.array([29780, 29780])                  # Velocidad inicial de la Tierra
v_inicial2 = np.array([0, 24100])                  # Velocidad inicial de Marte
v_inicial3 = np.array([0, 0])                      # Velocidad inicial del Sol (suponemos que está en reposo)


#SISTEMA CAOTICO (CREO QUE NO ESTA BIEN)
# Masas (en kg) — elegimos tres masas similares para aumentar el caos en el sistema
m1 = 5.972e24     # Masa del cuerpo 1
m2 = 4.867e24     # Masa del cuerpo 2
m3 = 3.285e24     # Masa del cuerpo 3

# Posiciones iniciales (en metros) — colocamos los cuerpos en posiciones que no forman una línea recta
posCuerpo1 = np.array([1.0e11, 0])                # Posición inicial del cuerpo 1
posCuerpo2 = np.array([-0.5e11, 0.866e11])        # Posición inicial del cuerpo 2 (colocado en una posición desfasada)
posCuerpo3 = np.array([-0.5e11, -0.866e11])       # Posición inicial del cuerpo 3 (otra posición desfasada)

# Velocidades iniciales (en m/s) — velocidades que apuntan en diferentes direcciones
v_inicial1 = np.array([0, 25000])                 # Velocidad inicial del cuerpo 1
v_inicial2 = np.array([-20000, -10000])           # Velocidad inicial del cuerpo 2
v_inicial3 = np.array([20000, -10000])            # Velocidad inicial del cuerpo 3



#INICIALIZACION DE VECTORES TRAYECTORIA, VELOCIDAD Y ENERGIA
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

#PARAMETROS TEMPORALES PARA EL BUCLE

t=0 #tiempo inicial
dt=60 #segundos en un minuto
t_max = 31536000 #segundos en un año
temp = np.linspace(0,dt,t_max) #se usa para la simulacion

while t<(t_max):
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
    
energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)
   
# Asegúrate de que las trayectorias son listas de numpy arrays
trayectoriaCuerpo1 = np.array(trayectoriaCuerpo1)
trayectoriaCuerpo2 = np.array(trayectoriaCuerpo2)
trayectoriaCuerpo3 = np.array(trayectoriaCuerpo3)

from matplotlib.animation import FuncAnimation

# Transponer trayectorias para obtener forma (525600, 2)
trayectoriaCuerpo1 = trayectoriaCuerpo1
trayectoriaCuerpo2 = trayectoriaCuerpo2
trayectoriaCuerpo3 = trayectoriaCuerpo3

# Crear la figura y los ejes

plt.switch_backend('TkAgg') #modificar donde aparece la grafica

fig, ax = plt.subplots(figsize=(8, 8))  # Tamaño de la figura en pulgadas
ax.set_aspect('equal')
ax.set_title('Movimiento de los Tres Cuerpos')

# Definir límites basados en tus datos
ax.set_xlim(-2e11, 2e11)
ax.set_ylim(-2e11, 2e11)

# Inicializar los puntos de los cuerpos
cuerpo1, = ax.plot([], [], 'ro', label="Cuerpo 1 (m1)")
cuerpo2, = ax.plot([], [], 'bo', label="Cuerpo 2 (m2)")
cuerpo3, = ax.plot([], [], 'go', label="Cuerpo 3 (m3)")

# Inicializar las líneas de las trayectorias
linea1, = ax.plot([], [], 'r-', alpha=0.5)
linea2, = ax.plot([], [], 'b-', alpha=0.5)
linea3, = ax.plot([], [], 'g-', alpha=0.5)

# Función para actualizar cada cuadro de la animación
def actualizar(frame):
    frame = min(frame, trayectoriaCuerpo1.shape[0] - 1)
    # Actualizar posiciones de los puntos usando listas para evitar la advertencia
    cuerpo1.set_data([trayectoriaCuerpo1[frame, 0]], [trayectoriaCuerpo1[frame, 1]])
    cuerpo2.set_data([trayectoriaCuerpo2[frame, 0]], [trayectoriaCuerpo2[frame, 1]])
    cuerpo3.set_data([trayectoriaCuerpo3[frame, 0]], [trayectoriaCuerpo3[frame, 1]])

    # Actualizar las líneas de las trayectorias
    linea1.set_data(trayectoriaCuerpo1[:frame+1, 0], trayectoriaCuerpo1[:frame+1, 1])
    linea2.set_data(trayectoriaCuerpo2[:frame+1, 0], trayectoriaCuerpo2[:frame+1, 1])
    linea3.set_data(trayectoriaCuerpo3[:frame+1, 0], trayectoriaCuerpo3[:frame+1, 1])

    return cuerpo1, cuerpo2, cuerpo3, linea1, linea2, linea3

# Crear la animación
frames = len(temp)
anim = FuncAnimation(fig, actualizar, frames=range(0, frames,100), interval=1, blit=True)

# Mostrar la animación
# Configurar la posición de la ventana
manager = plt.get_current_fig_manager()
manager.window.wm_geometry("+1000+50")  # Cambia los valores para ajustar la posición (x, y)

plt.legend()
plt.show()


"""
# Configuración de la gráfica
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

print("Energia acumulativa del sistema: ",  sum(eTotal) * dt)

print(f"Energía acumulada (Trapecio): {energia_acumulada_trapecio:.2f}")
print(f"Energía acumulada (Newton-Cotes): {energia_acumulada_newton_cotes:.2f}")
print(f"Energía acumulada (Cuadratura de Gauss): {energia_acumulada_gauss:.2f}")
"""
