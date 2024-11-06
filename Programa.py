import numpy as np
import matplotlib.pyplot as plt

def fuerzaGravitatoria(m1, m2, m3, pos1, pos2, pos3):
    G = 6.67e-11
    f_Grav1 = -G*m1*((m2*(pos2-pos1)/(np.linalg.norm(pos2-pos1)**3))+(m3*(pos3-pos1)/(np.linalg.norm(pos3-pos1)**3)))
    f_Grav2 = -G*m2*((m1*(pos1-pos2)/(np.linalg.norm(pos1-pos2)**3))+(m3*(pos3-pos2)/(np.linalg.norm(pos3-pos2)**3)))
    f_Grav3 = -G*m3*((m1*(pos1-pos3)/(np.linalg.norm(pos1-pos3)**3))+(m2*(pos2-pos3)/(np.linalg.norm(pos2-pos3)**3)))
    return f_Grav1, f_Grav2, f_Grav3

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
    
    f_grav1, f_grav2, f_grav3 = fuerzaGravitatoria(m1, m2, m3, pos1, pos2, pos3)
    
    # Retorna las derivadas [dx1/dt, dy1/dt, dvx1/dt, dvy1/dt, dx2/dt, dy2/dt, dvx2/dt, dvy2/dt, dx3/dt, dy3/dt, dvx3/dt, dvy3/dt]
    return np.array([
        vel1[0], vel1[1], f_grav1[0]/m1, f_grav1[1]/m1,
        vel2[0], vel2[1], f_grav2[0]/m2, f_grav2[1]/m2,
        vel3[0], vel3[1], f_grav3[0]/m3, f_grav3[1]/m3
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

def simular_3_cuerpos(m1, m2, m3, pos1_init, pos2_init, pos3_init, v1_init, v2_init, v3_init, t_final, dt):
    # Estado inicial
    estado = np.concatenate([pos1_init, v1_init, pos2_init, v2_init, pos3_init, v3_init])
    
    # Inicializar listas para almacenar las trayectorias
    t = 0
    times = [t]
    estados = [estado]
    
    while t < t_final:
        estado = rkf45_paso(estado, t, dt, m1, m2, m3)
        t += dt
        times.append(t)
        estados.append(estado)
    
    estados = np.array(estados)
    return times, estados

# Condiciones iniciales
m1 = 1.98e30
m2 = 5.97e24
m3 = 7645.80

posCuerpo1 = np.array([0, 0])
posCuerpo2 = np.array([5.0, 0])
posCuerpo3 = np.array([2.5,4.33])

v_inicial1 = np.array([0, 0])
v_inicial2 = np.array([0, 0])
v_inicial3 = np.array([0.0, 0.0])

# Simular
t_final = 50
dt = 0.1
tiempos, estados = simular_3_cuerpos(m1, m2, m3, posCuerpo1, posCuerpo2, posCuerpo3, 
                                  v_inicial1, v_inicial2, v_inicial3, t_final, dt)

# Extraer las trayectorias
trayectoriaCuerpo1 = estados[:, 0:2]
trayectoriaCuerpo2 = estados[:, 4:6]
trayectoriaCuerpo3 = estados[:, 8:10]

from matplotlib.animation import FuncAnimation

# Verifica las dimensiones de las trayectorias
if len(trayectoriaCuerpo1.shape) < 2 or trayectoriaCuerpo1.shape[1] < 2:
    raise ValueError("TrayectoriaCuerpo1 no tiene las dimensiones correctas.")
if len(trayectoriaCuerpo2.shape) < 2 or trayectoriaCuerpo2.shape[1] < 2:
    raise ValueError("TrayectoriaCuerpo2 no tiene las dimensiones correctas.")
if len(trayectoriaCuerpo3.shape) < 2 or trayectoriaCuerpo3.shape[1] < 2:
    raise ValueError("TrayectoriaCuerpo3 no tiene las dimensiones correctas.")

# Función para actualizar cada cuadro de la animación
def actualizar(frame):
    cuerpo1.set_data([trayectoriaCuerpo1[frame, 0]], [trayectoriaCuerpo1[frame, 1]])
    cuerpo2.set_data([trayectoriaCuerpo2[frame, 0]], [trayectoriaCuerpo2[frame, 1]])
    cuerpo3.set_data([trayectoriaCuerpo3[frame, 0]], [trayectoriaCuerpo3[frame, 1]])
    return cuerpo1, cuerpo2, cuerpo3

# Crear la figura y los ejes
fig, ax = plt.subplots()
max_x = max(np.max(trayectoriaCuerpo1[:, 0]), np.max(trayectoriaCuerpo2[:, 0]), np.max(trayectoriaCuerpo3[:, 0]))
min_x = min(np.min(trayectoriaCuerpo1[:, 0]), np.min(trayectoriaCuerpo2[:, 0]), np.min(trayectoriaCuerpo3[:, 0]))
max_y = max(np.max(trayectoriaCuerpo1[:, 1]), np.max(trayectoriaCuerpo2[:, 1]), np.max(trayectoriaCuerpo3[:, 1]))
min_y = min(np.min(trayectoriaCuerpo1[:, 1]), np.min(trayectoriaCuerpo2[:, 1]), np.min(trayectoriaCuerpo3[:, 1]))

ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

ax.set_aspect('equal')
ax.set_title('Movimiento de los Tres Cuerpos')

# Inicializar los puntos de los cuerpos
cuerpo1, = ax.plot([], [], 'ro', label="Cuerpo 1 (m1)")
cuerpo2, = ax.plot([], [], 'bo', label="Cuerpo 2 (m2)")
cuerpo3, = ax.plot([], [], 'go', label="Cuerpo 3 (m3)")

# Crear la animación
frames = len(tiempos)  # Número de cuadros en la animación
anim = FuncAnimation(fig, actualizar, frames=frames, interval=100, blit=True)

# Mostrar la animación
plt.legend()
plt.show()




""" 
# Graficar
plt.figure(figsize=(10, 10))
plt.xlim(-10, 10)
plt.ylim(-10, 10)

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
"""