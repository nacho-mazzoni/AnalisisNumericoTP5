import numpy as np
import matplotlib.pyplot as plt

def fuerzaGravitatoria(m1, m2, m3, pos1, pos2, pos3):
    G = 6.67e-11
    f_Grav1 = -G*m1*((m2*(pos2-pos1)/(np.linalg.norm(pos2-pos1)**3))+(m3*(pos3-pos1)/(np.linalg.norm(pos3-pos1)**3)))
    f_Grav2 = -G*m2*((m1*(pos1-pos2)/(np.linalg.norm(pos1-pos2)**3))+(m3*(pos3-pos2)/(np.linalg.norm(pos3-pos2)**3)))
    f_Grav3 = -G*m3*((m1*(pos1-pos3)/(np.linalg.norm(pos1-pos3)**3))+(m2*(pos2-pos3)/(np.linalg.norm(pos2-pos3)**3)))
    return f_Grav1, f_Grav2, f_Grav3

def aceleracionGravitatoria(m1, m2, m3, pos1, pos2, pos3):
    G = 6.67e-11
    masas = np.array([m1, m2, m3])
    posiciones = np.array([pos1, pos2, pos3])
    aceleraciones = np.zeros_like(posiciones)
    cuerpos = 3
    for i in range(cuerpos):
        for j in range(cuerpos):
            if j!=i:
                r = posiciones[j]-posiciones[i]
                dist = np.linalg.norm(r)
                if dist>0:
                    aceleraciones[i] += G*masas[j]*r/(dist**3)
    return aceleraciones

def derivar(estado, t, m1, m2, m3):
    """
    Calcula las derivadas del estado para el problema de los tres cuerpos.
    estado: [pos1_x, pos1_y, vel1_x, vel1_y, pos2_x, pos2_y, vel2_x, vel2_y, pos3_x, pos3_y, vel3_x, vel3_y]
    """
    pos1 = np.array([estado[0], estado[1]])
    pos2 = np.array([estado[4], estado[5]])
    pos3 = np.array([estado[8], estado[9]])
    
    vel1 = np.array([estado[2], estado[3]])
    vel2 = np.array([estado[6], estado[7]])
    vel3 = np.array([estado[10], estado[11]])
    
    aceleraciones = aceleracionGravitatoria(m1, m2, m3, pos1, pos2, pos3)
    aceleracionCuerpo1 = aceleraciones[0]
    aceleracionCuerpo2 = aceleraciones[1]
    aceleracionCuerpo3 = aceleraciones[2]
    # Retorna las derivadas [dx1/dt, dy1/dt, dvx1/dt, dvy1/dt, dx2/dt, dy2/dt, dvx2/dt, dvy2/dt, dx3/dt, dy3/dt, dvx3/dt, dvy3/dt]
    return np.array([
        vel1[0], vel1[1], aceleracionCuerpo1[0], aceleracionCuerpo1[1],
        vel2[0], vel2[1], aceleracionCuerpo2[0], aceleracionCuerpo2[1],
        vel3[0], vel3[1], aceleracionCuerpo3[0], aceleracionCuerpo3[1]
    ])

def rkf45_paso(estado, t, h, m1, m2, m3):
    """
    Implementa un paso del método RKF45
    """
    # Coeficientes RKF45
    a2, a3, a4, a5, a6 = 1/4, 3/8, 12/13, 1, 1/2
    
    b21 = 1/4
    b31, b32 = 3/32, 9/32
    b41, b42, b43 = 1932/2197, -7200/2197, 7296/2197
    b51, b52, b53, b54 = 439/216, -8, 3680/513, -845/4104
    b61, b62, b63, b64, b65 = -8/27, 2, -3544/2565, 1859/4104, -11/40
    
    # Coeficientes para el método de orden 5
    c1, c3, c4, c5 = 16/135, 6656/12825, 28561/56430, -9/50
    
    # Calcular los k's
    k1 = h * derivar(estado, t, m1, m2, m3)
    k2 = h * derivar(estado + b21*k1, t + a2*h, m1, m2, m3)
    k3 = h * derivar(estado + b31*k1 + b32*k2, t + a3*h, m1, m2, m3)
    k4 = h * derivar(estado + b41*k1 + b42*k2 + b43*k3, t + a4*h, m1, m2, m3)
    k5 = h * derivar(estado + b51*k1 + b52*k2 + b53*k3 + b54*k4, t + h, m1, m2, m3)
    k6 = h * derivar(estado + b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5, t + a6*h, m1, m2, m3)
    
    # Calcular el nuevo estado (orden 5)
    new_estado = estado + c1*k1 + c3*k3 + c4*k4 + c5*k5
    
    return new_estado

def rk3_paso(estado, t, h, m1, m2, m3):
    #coeficientes
    k1 = h * derivar(estado, t, m1, m2, m3)
    k2 = h * derivar(estado + h*0.5, t + k1*0.5, m1, m2, m3)
    K_prima = h * derivar(estado + h, t + k1, m1, m2, m3)
    k3 = h * derivar(estado + h, t + K_prima, m1, m2, m3)
    
    nuevo_estado = estado + (1/6)*(k1 + 4*k2 +k3)

    return nuevo_estado

def estimadorDeError(y3, y6, r):
    error = (y3-y6/(2**r)-1)-2**r
    return error

#arrrglos con energias cineticas y potenciales para cada cuerpo
eCineticaCuerpo1 = []
eCineticaCuerpo2 = []
eCineticaCuerpo3 = []
ePotGravitacional1 = []
ePotGravitacional2 = []
ePotGravitacional3 = []
eTotal = []

estimadorDelError = []

def simular_3_cuerpos(m1, m2, m3, pos1_init, pos2_init, pos3_init, v1_init, v2_init, v3_init, t_max, dt):
    # Estado inicial
    estado = np.concatenate([pos1_init, v1_init, pos2_init, v2_init, pos3_init, v3_init])
    estado3 = np.concatenate([pos1_init, v1_init, pos2_init, v2_init, pos3_init, v3_init])
    
    # Inicializar listas para almacenar las trayectorias
    t = 0
    times = [t]
    estados = [estado]
    
    while t < t_max:
        estimadorDelError.append(estimadorDeError(estado3, estado, 6))
        estado = rkf45_paso(estado, t, dt, m1, m2, m3)
        estado3 = rkf45_paso(estado3, t, dt*0.5, m1, m2, m3)
        t += dt
        times.append(t)
        estados.append(estado)
        e_cinetica1, e_cinetica2, e_cinetica3 = energiaCinetica(m1, m2, m3, estado[1], estado[3], estado[5])
        eCineticaCuerpo1.append(e_cinetica1)
        eCineticaCuerpo2.append(e_cinetica2)
        eCineticaCuerpo3.append(e_cinetica3)
        e_potencial1, e_potencial2, e_potencial3 = energiaPotencialGrav(m1, m2, m3, estado[0], estado[2], estado[4])
        ePotGravitacional1.append(e_potencial1)
        ePotGravitacional2.append(e_potencial2)
        ePotGravitacional3.append(e_potencial3)
        eTotal.append(e_cinetica1+e_cinetica2+e_cinetica3+e_potencial1+e_potencial2+e_potencial3)
    
    estados = np.array(estados)
    return times, estados

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
dt=60 #segundos en un minuto
t_max = (31536000*2) #segundos en un año
tiempos, estados = simular_3_cuerpos(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3, 
                                  v_inicial1, v_inicial2, v_inicial3, t_max, dt)

# Extraer las trayectorias
trayectoriaCuerpo1 = estados[:, 0:2]
trayectoriaCuerpo2 = estados[:, 4:6]
trayectoriaCuerpo3 = estados[:, 8:10]

# Graficar
plt.figure(figsize=(10, 10))
plt.xlim(-2e11, 2e11)  # Ajusta estos límites según tus necesidades
plt.ylim(-2e11, 2e11)

plt.plot(trayectoriaCuerpo1[:, 0], trayectoriaCuerpo1[:, 1], 'b-', label='Cuerpo 1')
plt.plot(trayectoriaCuerpo2[:, 0], trayectoriaCuerpo2[:, 1], 'r-', label='Cuerpo 2')
plt.plot(trayectoriaCuerpo3[:, 0], trayectoriaCuerpo3[:, 1], 'g-', label='Cuerpo 3')

plt.plot(trayectoriaCuerpo1[-1, 0], trayectoriaCuerpo1[-1, 1], 'bo')
plt.plot(trayectoriaCuerpo2[-1, 0], trayectoriaCuerpo2[-1, 1], 'ro')
plt.plot(trayectoriaCuerpo3[-1, 0], trayectoriaCuerpo3[-1, 1], 'go')

plt.title('Trayectorias de los Cuerpos (RKF45)')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)

energia_acumulada_trapecio = trapecio(eTotal, dt)
energia_acumulada_newton_cotes = newton_cotes(eTotal, dt)
energia_acumulada_gauss = gauss_quadrature(eTotal, dt)

print("Energia acumulativa del sistema: ",  sum(eTotal) * dt)

print(f"Energía acumulada (Trapecio): {energia_acumulada_trapecio:.2f}")
print(f"Energía acumulada (Newton-Cotes): {energia_acumulada_newton_cotes:.2f}")
print(f"Energía acumulada (Cuadratura de Gauss): {energia_acumulada_gauss:.2f}")
print("Errores estimados: ", estimadorDelError)