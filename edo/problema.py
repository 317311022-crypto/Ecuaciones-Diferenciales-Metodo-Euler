"""Definición de un Problema de Valor Inicial (PVI).

Un PVI de primer orden tiene la forma

    y'(t) = f(t, y),    y(t0) = y0,    t en [t0, t_final]

``y`` puede ser un escalar (una sola ecuación) o un vector (un sistema de
ecuaciones). Las ecuaciones de orden superior se reescriben como sistemas
de primer orden (ver ``edo.ejemplos.oscilador_armonico``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

FuncionEDO = Callable[[float, np.ndarray], np.ndarray]
FuncionExacta = Callable[[np.ndarray], np.ndarray]


@dataclass
class ProblemaValorInicial:
    """Agrupa todo lo necesario para describir un PVI.

    Parámetros
    ----------
    f : callable
        Lado derecho de la EDO, ``f(t, y)``.
    t0 : float
        Tiempo inicial.
    y0 : float o array
        Condición inicial ``y(t0)``.
    t_final : float
        Tiempo final del intervalo de integración.
    solucion_exacta : callable, opcional
        Solución analítica ``y(t)`` si se conoce. Se usa para medir el error.
    nombre : str
        Nombre corto del problema (para títulos de gráficas).
    ecuacion : str
        Representación en texto de la ecuación, p. ej. ``"y' = y - t^2 + 1"``.
    """

    f: FuncionEDO
    t0: float
    y0: float | np.ndarray
    t_final: float
    solucion_exacta: Optional[FuncionExacta] = None
    nombre: str = "PVI"
    ecuacion: str = ""
    etiquetas: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.t0 = float(self.t0)
        self.t_final = float(self.t_final)
        self.y0 = np.asarray(self.y0, dtype=float)
        if self.y0.ndim > 1:
            raise ValueError("y0 debe ser un escalar o un vector (1-D).")
        if self.t_final <= self.t0:
            raise ValueError("t_final debe ser mayor que t0.")

    @property
    def es_sistema(self) -> bool:
        """``True`` si el problema es un sistema de ecuaciones."""
        return self.y0.ndim == 1

    @property
    def dimension(self) -> int:
        """Número de ecuaciones del problema."""
        return 1 if self.y0.ndim == 0 else self.y0.shape[0]

    @property
    def tiene_solucion_exacta(self) -> bool:
        return self.solucion_exacta is not None

    def exacta(self, t: np.ndarray) -> np.ndarray:
        """Evalúa la solución exacta en los tiempos ``t``."""
        if self.solucion_exacta is None:
            raise ValueError(f"El problema '{self.nombre}' no tiene solución exacta.")
        return np.asarray(self.solucion_exacta(np.asarray(t, dtype=float)), dtype=float)

    def __str__(self) -> str:
        ec = self.ecuacion or "y' = f(t, y)"
        if self.es_sistema:
            var = "(" + ", ".join(self.etiquetas) + ")" if self.etiquetas else "y"
            y0 = "(" + ", ".join(f"{v:g}" for v in self.y0) + ")"
        else:
            var = self.etiquetas[0] if self.etiquetas else "y"
            y0 = f"{float(self.y0):g}"
        return f"{self.nombre}: {ec},  {var}({self.t0:g}) = {y0},  t ∈ [{self.t0:g}, {self.t_final:g}]"
