import numpy as np
def derivs(x, y):
    # Ejemplo de derivada: dy/dx = -y (decay exponencial)
    return -y
def RKkc(y, dy, x, h):
    # Definir los parámetros constantes
    a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875
    b21, b31, b32 = 0.2, 3/40, 9/40
    b41, b42, b43 = 0.3, -0.9, 1.2
    b51, b52, b53, b54 = -11/54, 2.5, -70/27, 35/27
    b61, b62, b63, b64, b65 = 1631/55296, 175/512, 575/13824, 44275/110592, 253/4096
    c1, c3, c4, c6 = 37/378, 250/621, 125/594, 512/1771
    dc1, dc3, dc4, dc5, dc6 = c1 - 2825/27648, c3 - 18575/48384, c4 - 13525/55296, -277/14336, c6 - 0.25
    
    # Realizar los cálculos para cada paso de k
    ytemp = y + b21 * h * dy
    k2 = derivs(x + a2 * h, ytemp)
    
    ytemp = y + h * (b31 * dy + b32 * k2)
    k3 = derivs(x + a3 * h, ytemp)
    
    ytemp = y + h * (b41 * dy + b42 * k2 + b43 * k3)
    k4 = derivs(x + a4 * h, ytemp)
    
    ytemp = y + h * (b51 * dy + b52 * k2 + b53 * k3 + b54 * k4)
    k5 = derivs(x + a5 * h, ytemp)
    
    ytemp = y + h * (b61 * dy + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
    k6 = derivs(x + a6 * h, ytemp)
    
    # Calcular yout y yerr
    yout = y + h * (c1 * dy + c3 * k3 + c4 * k4 + c6 * k6)
    yerr = h * (dc1 * dy + dc3 * k3 + dc4 * k4 + dc5 * k5 + dc6 * k6)
    
    return yout, yerr
def Adapt(x, y, dy, htry, yscal, eps):
    safety, econ = 0.9, 1.89e-4
    h = htry
    
    while True:
        ytemp, yerr = RKkc(y, dy, x, h)
        emax = max(abs(yerr / (yscal * eps)))
        
        if emax <= 1:
            break
        htemp = safety * h * emax**-0.25
        h = max(abs(htemp), 0.25 * abs(h))
        xnew = x + h
        
        if xnew == x:
            raise RuntimeError("Error: Paso demasiado pequeño en Adapt")
    
    if emax > econ:
        hnxt = safety * emax**-0.2 * h
    else:
        hnxt = 4.0 * h
    
    return x + h, ytemp, hnxt
