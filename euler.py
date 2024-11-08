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
# masas en kg (Tierra, Luna, Sol)
m1 = 5.97e24
m2 = 7.348e22
m3 = 1.98e30  

#Posiciones iniciales aproximadas
posCuerpo1 = np.array([1.4961e11, 0])   # Tierra
posCuerpo2 = np.array([1.4961e11*np.cos(np.pi/3), 1.4961e11*np.sin(np.pi/3)])  # Luna
posCuerpo3 = np.array([0, 0])                     # Sol

#Velocidades iniciales aproximadas
v_inicial1 = np.array([0, 29780])
v_inicial2 = np.array([0, -30802])
v_inicial3 = np.array([0, 0])

#Tiempo inicial, dt y tiempo maximo de iteracion
t=0 #tiempo inicial
dt=60 #segundos en un minuto
t_max = (31536000*2) #segundos en un año

#CALCULO DEL CENTRO DE MASA
centro_de_masasX = (m1 * posCuerpo1[0] + m2 * posCuerpo2[0] + m3 * posCuerpo3[0]) / (m1 + m2 + m3)
centro_de_masasY = (m1 * posCuerpo1[1] + m2 * posCuerpo2[1] + m3 * posCuerpo3[1]) / (m1 + m2 + m3)
centroDeMasa = np.array([centro_de_masasX, centro_de_masasY])


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
    

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)
print(f"Energía acumulada (Trapecio): {energia_acumulada_trapecio:.2f}")
print(f"Energía acumulada (Newton-Cotes): {energia_acumulada_newton_cotes:.2f}")
print(f"Energía acumulada (Cuadratura de Gauss): {energia_acumulada_gauss:.2f}")

# Asegúrate de que las trayectorias son listas de numpy arrays
trayectoriaCuerpo1 = np.array(trayectoriaCuerpo1)  # Suponiendo que ya tienes las trayectorias calculadas
trayectoriaCuerpo2 = np.array(trayectoriaCuerpo2)
trayectoriaCuerpo3 = np.array(trayectoriaCuerpo3)

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
