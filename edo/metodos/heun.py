"""Método de Euler mejorado (Heun).

Primero se da un paso de Euler para *predecir* ``y_{n+1}``; después se
*corrige* usando el promedio de las pendientes al inicio y al final del paso.

    k1 = f(t_n, y_n)
    k2 = f(t_n + h, y_n + h·k1)
    y_{n+1} = y_n + (h/2)(k1 + k2)

Orden 2.
"""

from __future__ import annotations

import numpy as np

from ..formato import num, nump, sub
from ..problema import FuncionEDO
from .base import MetodoNumerico, Paso
from .registro import registrar


@registrar
class Heun(MetodoNumerico):
    nombre = "heun"
    nombre_completo = "Euler mejorado (Heun)"
    orden = 2
    formula = "yₙ₊₁ = yₙ + (h/2)·(k₁ + k₂)"

    def paso(self, f: FuncionEDO, t: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = self.evaluar(f, t, y, "k1")                  # pendiente al inicio
        k2 = self.evaluar(f, t + h, y + h * k1, "k2")     # pendiente en el punto predicho por Euler
        return y + (h / 2) * (k1 + k2)

    def funcion_estabilidad(self, z: complex) -> complex:
        return 1 + z + z**2 / 2

    def formula_numerica(self, paso: Paso) -> str | None:
        if paso.y.ndim or len(paso.etapas) != 2:
            return None
        n = paso.indice
        k1, k2 = (e.pendiente for e in paso.etapas)
        return (
            f"y{sub(n + 1)} = y{sub(n)} + (h/2)·(k₁ + k₂)\n"
            f"   = {num(paso.y)} + {num(paso.h / 2)}·({num(k1)} + {nump(k2)}) = {num(paso.y_siguiente)}"
        )
