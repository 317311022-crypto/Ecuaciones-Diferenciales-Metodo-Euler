"""Método de Euler (explícito).

Idea: en cada punto ``(t_n, y_n)`` avanzamos una distancia ``h`` sobre la
recta tangente a la solución. La pendiente de esa recta la da la propia
ecuación diferencial: ``y'(t_n) = f(t_n, y_n)``.

    y_{n+1} = y_n + h · f(t_n, y_n)

Se obtiene truncando la serie de Taylor de ``y(t_n + h)`` después del
término lineal, por lo que el error local es O(h²) y el error global O(h):
el método es de **orden 1** (si h se reduce a la mitad, el error también).

Este archivo es la plantilla para agregar métodos nuevos: copia la clase,
cambia ``nombre``, ``orden`` y el cuerpo de ``paso``.
"""

from __future__ import annotations

import numpy as np

from ..formato import num, nump, sub
from ..problema import FuncionEDO
from .base import MetodoNumerico, Paso
from .registro import registrar


@registrar
class Euler(MetodoNumerico):
    nombre = "euler"
    nombre_completo = "Euler"
    orden = 1
    formula = "yₙ₊₁ = yₙ + h·f(xₙ, yₙ)"

    def paso(self, f: FuncionEDO, t: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = self.evaluar(f, t, y, "k1")     # pendiente en el punto de partida
        return y + h * k1                    # avanzar h sobre la recta tangente

    def funcion_estabilidad(self, z: complex) -> complex:
        # y' = λy  →  y_{n+1} = y_n + hλ y_n = (1 + z) y_n
        return 1 + z

    def formula_numerica(self, paso: Paso) -> str | None:
        if paso.y.ndim or not paso.etapas:
            return None
        n, k1 = paso.indice, paso.etapas[0].pendiente
        return (
            f"y{sub(n + 1)} = y{sub(n)} + h·k₁ = {num(paso.y)} + {num(paso.h)}·{nump(k1)}"
            f" = {num(paso.y_siguiente)}"
        )
