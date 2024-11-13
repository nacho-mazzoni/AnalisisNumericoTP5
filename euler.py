import numpy as np
import matplotlib.pyplot as plt

#defino la fuerza gravitacional de cada cuerpo ejercida por los otros dos
def f_gravitacional(m1, m2, m3, pos1, pos2, pos3):
    G = 6.67e-11
    posiciones = np.array([pos1, pos2, pos3])
    masas = np.array([m1, m2, m3])
    fuerzasGravitatorias = np.zeros_like(posiciones)
    cuerpos=3
    fuerzaLimite = np.array([10000000, 1000000]) #para los casos en que los cuerpos se choquen, sale disparado 
    for i in range(cuerpos):
        for j in range(cuerpos):
            if i != j:
                dist = posiciones[j]-posiciones[i]
                if(np.linalg.norm(dist)!=0):
                    fuerzasGravitatorias[i] += (G) * masas[i] * (masas[j]*dist/(np.linalg.norm(dist)**3))
                else:
                    fuerzasGravitatorias[i] = fuerzaLimite 
    return fuerzasGravitatorias

#defino la funcion aceleracion Fgrav/mi
def f_aceleracion(fuerzasGravitatorias, m1 ,m2, m3):
    aceleraciones = np.zeros_like(fuerzasGravitatorias)
    masas = np.array([m1, m2, m3])
    for i in range(3):
        aceleraciones[i] = (fuerzasGravitatorias[i]/masas[i])
    return aceleraciones

#defino la funcion velocidad
""" deberia hacer la integral de t a t+h de la funcion gravitatoria, 
lo unico que cambiara es la posicion en funcion del tiempo"""

def f_Velocidad(m1, m2, m3, vC1, vC2, vC3, h, pos1, pos2, pos3):
    fuerzasGravitatorias = f_gravitacional(m1, m2, m3, pos1, pos2, pos3)
    aceleraciones = f_aceleracion(fuerzasGravitatorias, m1, m2, m3)
    v_1 = vC1 + aceleraciones[0] * h
    v_2 = vC2 + aceleraciones[1] * h
    v_3 = vC3 + aceleraciones[2] * h
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

def estimadorDeError(y3, y6):
    return y3-y6


# Condiciones iniciales
m1 = 5.97e24
m2 = 7645.80
m3 = 1.98e30

#Posiciones iniciales aproximadas
posCuerpo1 = np.array([1.4961e11, 0])   # Tierra
posCuerpo2 = np.array([7.4805e10, 1.2956e11])  # Luna
posCuerpo3 = np.array([0, 0])                     # Sol

#Velocidades iniciales aproximadas
v_inicial1 = np.array([0, 29780])
v_inicial2 = np.array([0, -30802])
v_inicial3 = np.array([0, 0])

#Tiempo inicial, dt y tiempo maximo de iteracion
t=0 #tiempo inicial
dt=30 #segundos en un minuto
t_max = (63072000) #segundos en un año
#salto = 0.1 #salto

#inicializar trayectorias
trayectoriaCuerpo1 = []
trayectoriaCuerpo2 = []
trayectoriaCuerpo3 = []

#inicializar velocidades
velocidades_cuerpo1 = []
velocidades_cuerpo2 = []
velocidades_cuerpo3 = []

#arreglos con energias cineticas y potenciales para cada cuerpo
eCineticaCuerpo1 = []
eCineticaCuerpo2 = []
eCineticaCuerpo3 = []
ePotGravitacional1 = []
ePotGravitacional2 = []
ePotGravitacional3 = []
eTotal = []
estimadorDelError = []

v_inicial13 = v_inicial1
v_inicial23 = v_inicial2
v_inicial33 = v_inicial3
posCuerpo13 = posCuerpo1
posCuerpo23 = posCuerpo2
posCuerpo33 = posCuerpo3

tiempos = []

#CAMBIOS PARA QUE SEA ADAPTATIVO
while t<t_max:
    #estado = np.concatenate([posCuerpo1, posCuerpo2, posCuerpo3])
    #estado_prima = np.concatenate([posCuerpo13, posCuerpo23, posCuerpo33])
    #estimadorDelError.append(estimadorDeError(estado_prima, estado))
    #calculo de la velocidad en funcion de la fuerza grav
    v_cuerpo1, v_cuerpo2, v_cuerpo3 = f_Velocidad(m1, m2, m3, v_inicial1, v_inicial2, v_inicial3, dt, posCuerpo1, posCuerpo2, posCuerpo3)

    #calculo de la posicion en funcion de la velocidad
    posNueva_cuerpo1, posNueva_cuerpo2, posNueva_cuerpo3 = f_posicion(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3, dt,
                                                              v_cuerpo1, v_cuerpo2, v_cuerpo3)

    #v_inicial13, v_inicial23, v_inicial33 = f_Velocidad(m1, m2, m3, v_inicial13, v_inicial23, v_inicial33, dt*0.5, posCuerpo13, posCuerpo23, posCuerpo33)

    #posCuerpo13, posCuerpo23, posCuerpo33 = f_posicion(m1, m2 , m3, posCuerpo13, posCuerpo23, posCuerpo33, dt*0.5, v_inicial13, v_inicial23, v_inicial33)    
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
    #if (np.abs(np.linalg.norm(estimadorDelError[t])) > 10e-5):
        #salto = salto*((10e-5/np.linalg.norm(estimadorDelError[t]))**0.5)
    #else:
        #salto *= 1.5
    tiempos.append(t)
    t += dt
    

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)

# Asegúrate de que las trayectorias son listas de numpy arrays
trayectoriaCuerpo1 = np.array(trayectoriaCuerpo1)  # Suponiendo que ya tienes las trayectorias calculadas
trayectoriaCuerpo2 = np.array(trayectoriaCuerpo2)
trayectoriaCuerpo3 = np.array(trayectoriaCuerpo3)
"""
# Configuración de la gráfica
plt.figure(figsize=(10, 10))
plt.xlim(-10e10, 10e10)  # Ajusta estos límites según tus necesidades
plt.ylim(-10e10, 10e10)

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
"""

print("Energia acumulativa del sistema: ",  sum(eTotal) * dt)
print("Energia Acumulada con la funcion: ", energiaAcumulativa(eTotal[0], eTotal[-1]))
print(f"Energía acumulada (Trapecio): {energia_acumulada_trapecio:.2f}")
print(f"Energía acumulada (Newton-Cotes): {energia_acumulada_newton_cotes:.2f}")
print(f"Energía acumulada (Cuadratura de Gauss): {energia_acumulada_gauss:.2f}")
#print("Errores estimados: ", estimadorDelError)

# Crear la figura y el gráfico
plt.figure(figsize=(10, 6))
plt.xlim(min(tiempos), max(tiempos))
plt.ylim(min(eTotal), max(eTotal))
# Graficar cada array en función del tiempo con colores distintos y una etiqueta para la leyenda
plt.plot(tiempos, eTotal, color='blue', label='Energia total por pasos')
# Agregar título y etiquetas
plt.title("Datos de la energía en función del tiempo")
plt.xlabel("Tiempo")
plt.ylabel("Energia")
# Mostrar la leyenda para diferenciar cada cuerpo
plt.legend()
# Mostrar la cuadrícula
plt.grid(True)
plt.show()