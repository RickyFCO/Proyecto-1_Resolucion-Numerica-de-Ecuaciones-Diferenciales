import numpy as np
import matplotlib.pyplot as plt

# === Definición de las ecuaciones diferenciales ===

# EDO Lineal de Primer Orden: dy/dx = y
def edo_lineal(x, y):
    return y

# Enfriamiento de Newton: dT/dt = -k(T - T_amb)
def enfriamiento_newton(t, T, k=0.1, T_amb=20):
    return -k * (T - T_amb)

# Crecimiento Logístico: dP/dt = rP(1 - P/K)
def crecimiento_logistico(t, P, r=0.5, K=100):
    return r * P * (1 - P / K)

# Movimiento Armónico Simple: d²x/dt² = -ω²x
# Se transforma en un sistema de primer orden.
def movimiento_armonico_simple(t, X, omega=2):
    x, v = X
    dxdt = v
    dvdt = -omega**2 * x
    return np.array([dxdt, dvdt])

# === Métodos Numéricos ===

# Método de Euler
def metodo_euler(f, x0, y0, h, n, is_system=False):
    x = [x0]
    y = [y0]
    for i in range(n):
        y_next = y[-1] + h * f(x[-1], y[-1]) if not is_system else y[-1] + h * np.array(f(x[-1], y[-1]))
        x_next = x[-1] + h
        x.append(x_next)
        y.append(y_next)
    return x, y

# Método de Runge-Kutta de Cuarto Orden (RK4)
def metodo_rk4(f, x0, y0, h, n, is_system=False):
    x = [x0]
    y = [y0]
    for i in range(n):
        k1 = f(x[-1], y[-1])
        k2 = f(x[-1] + h/2, y[-1] + h/2 * k1) if not is_system else f(x[-1] + h/2, y[-1] + h/2 * np.array(k1))
        k3 = f(x[-1] + h/2, y[-1] + h/2 * k2) if not is_system else f(x[-1] + h/2, y[-1] + h/2 * np.array(k2))
        k4 = f(x[-1] + h, y[-1] + h * k3) if not is_system else f(x[-1] + h, y[-1] + h * np.array(k3))
        
        y_next = y[-1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4) if not is_system else y[-1] + (h / 6) * np.array(k1 + 2*k2 + 2*k3 + k4)
        x_next = x[-1] + h
        
        x.append(x_next)
        y.append(y_next)
    return x, y

# === Soluciones Analíticas para Validación ===
def solucion_analitica_lineal(x):
    return np.exp(x)

def solucion_enfriamiento_newton(t, T0=100, k=0.1, T_amb=20):
    return T_amb + (T0 - T_amb) * np.exp(-k * t)

def solucion_crecimiento_logistico(t, P0=10, r=0.5, K=100):
    return K / (1 + (K - P0) / P0 * np.exp(-r * t))

def solucion_movimiento_armonico_simple(t, A=1, omega=2):
    return A * np.cos(omega * t)

# === Pruebas y Gráficas ===
def graficar_resultados(x, y, x_exacta, y_exacta, titulo, xlabel='x', ylabel='y'):
    plt.plot(x_exacta, y_exacta, label='Solución Analítica', color='green')
    plt.plot(x, y, 'o-', label='Solución Numérica', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.show()

# Parámetros comunes
x0, y0 = 0, 1
h = 0.1
n = 20

# Prueba: EDO Lineal de Primer Orden
x_euler, y_euler = metodo_euler(edo_lineal, x0, y0, h, n)
x_rk4, y_rk4 = metodo_rk4(edo_lineal, x0, y0, h, n)

# Solución analítica para comparación
x_exacta = np.linspace(0, h * n, 100)
y_exacta = solucion_analitica_lineal(x_exacta)

graficar_resultados(x_euler, y_euler, x_exacta, y_exacta, 'Método de Euler vs Solución Analítica')
graficar_resultados(x_rk4, y_rk4, x_exacta, y_exacta, 'Método de RK4 vs Solución Analítica')

# Puedes incluir más pruebas con otras ecuaciones diferenciales y métodos aquí.
