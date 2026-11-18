"""Euler·Lab — método de Euler paso a paso.

Escribe aquí arriba la ecuación diferencial y los parámetros, y ejecuta:

    python euler_app.py

Todo se puede cambiar después desde la barra superior de la ventana
(Enter o "▶ Resolver" para recalcular).
"""

# ╔══════════════════════════════ CONFIGURACIÓN ══════════════════════════════╗
ECUACION = "x + y"              # y' = f(x, y)   ej.: "y(1 - y/10)", "sin(x) - y", "-15y"
X0 = 0                          # condición inicial: y(X0) = Y0
Y0 = 1
H = 0.1                         # tamaño de paso
X_FINAL = 2                     # integrar hasta aquí
METODO = "euler"                # "euler", "heun", "rk4" o cualquier método registrado
EXACTA = "2*exp(x) - x - 1"     # solución exacta y(x) para medir el error ("" si no se conoce:
                                # entonces se usa RK4 con un paso muy fino como referencia)
CALIDAD = "media"               # "media": laptops (fondo fijo, sin parallax) · "alta": todos los
                                # efectos con parallax (PC de escritorio) · "baja": equipos lentos
# ╚═══════════════════════════════════════════════════════════════════════════╝

from edo import Configuracion
from edo.app import main

if __name__ == "__main__":
    main(Configuracion(ECUACION, X0, Y0, H, X_FINAL, METODO, EXACTA), calidad=CALIDAD)
