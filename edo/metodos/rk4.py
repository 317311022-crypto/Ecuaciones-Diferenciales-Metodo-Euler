"""Método de Runge-Kutta clásico de orden 4 (RK4).

    k1 = f(t_n,       y_n)
    k2 = f(t_n + h/2, y_n + h/2·k1)
    k3 = f(t_n + h/2, y_n + h/2·k2)
    k4 = f(t_n + h,   y_n + h·k3)
    y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)

Orden 4. Se incluye como referencia para comparar contra Euler.
"""

from __future__ import annotations

import numpy as np

from ..formato import num, nump, sub
from ..problema import FuncionEDO
from .base import MetodoNumerico, Paso
from .registro import registrar


@registrar
class RungeKutta4(MetodoNumerico):
    nombre = "rk4"
    nombre_completo = "Runge-Kutta 4"
    orden = 4
    formula = "yₙ₊₁ = yₙ + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)"

    def paso(self, f: FuncionEDO, t: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = self.evaluar(f, t, y, "k1")
        k2 = self.evaluar(f, t + h / 2, y + h / 2 * k1, "k2")
        k3 = self.evaluar(f, t + h / 2, y + h / 2 * k2, "k3")
        k4 = self.evaluar(f, t + h, y + h * k3, "k4")
        return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def funcion_estabilidad(self, z: complex) -> complex:
        return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24

    def formula_numerica(self, paso: Paso) -> str | None:
        if paso.y.ndim or len(paso.etapas) != 4:
            return None
        n = paso.indice
        k1, k2, k3, k4 = (e.pendiente for e in paso.etapas)
        return (
            f"y{sub(n + 1)} = y{sub(n)} + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)\n"
            f"   = {num(paso.y)} + {num(paso.h / 6)}·({num(k1)} + 2·{nump(k2)} + 2·{nump(k3)} + {nump(k4)})\n"
            f"   = {num(paso.y_siguiente)}"
        )
