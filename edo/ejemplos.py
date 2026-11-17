"""Problemas de valor inicial clásicos, todos con solución exacta conocida
para poder medir el error de los métodos numéricos."""

from __future__ import annotations

import numpy as np

from .problema import ProblemaValorInicial


def crecimiento_exponencial(k: float = 1.0, y0: float = 1.0, t_final: float = 2.0) -> ProblemaValorInicial:
    """y' = k·y,  y(0) = y0.   Solución: y(t) = y0·e^{kt}."""
    return ProblemaValorInicial(
        f=lambda t, y: k * y,
        t0=0.0,
        y0=y0,
        t_final=t_final,
        solucion_exacta=lambda t: y0 * np.exp(k * t),
        nombre="Crecimiento exponencial",
        ecuacion=f"y' = {k:g}·y",
    )


def ejemplo_burden() -> ProblemaValorInicial:
    """y' = y - t² + 1,  y(0) = 0.5,  t ∈ [0, 2].

    Ejemplo clásico de Burden & Faires, *Análisis Numérico*.
    Solución: y(t) = (t + 1)² - 0.5·e^t.
    """
    return ProblemaValorInicial(
        f=lambda t, y: y - t**2 + 1,
        t0=0.0,
        y0=0.5,
        t_final=2.0,
        solucion_exacta=lambda t: (t + 1) ** 2 - 0.5 * np.exp(t),
        nombre="Ejemplo de Burden",
        ecuacion="y' = y - t² + 1",
    )


def enfriamiento_newton(
    k: float = 0.1, temperatura_ambiente: float = 20.0, T0: float = 90.0, t_final: float = 30.0
) -> ProblemaValorInicial:
    """Ley de enfriamiento de Newton: T' = -k (T - Ta),  T(0) = T0.

    Solución: T(t) = Ta + (T0 - Ta)·e^{-kt}.
    """
    Ta = temperatura_ambiente
    return ProblemaValorInicial(
        f=lambda t, T: -k * (T - Ta),
        t0=0.0,
        y0=T0,
        t_final=t_final,
        solucion_exacta=lambda t: Ta + (T0 - Ta) * np.exp(-k * t),
        nombre="Enfriamiento de Newton",
        ecuacion=f"T' = -{k:g}(T - {Ta:g})",
        etiquetas=("T",),
    )


def logistica(r: float = 1.0, K: float = 10.0, y0: float = 1.0, t_final: float = 10.0) -> ProblemaValorInicial:
    """Crecimiento logístico: y' = r·y(1 - y/K).

    Solución: y(t) = K / (1 + ((K - y0)/y0)·e^{-rt}).
    """
    return ProblemaValorInicial(
        f=lambda t, y: r * y * (1 - y / K),
        t0=0.0,
        y0=y0,
        t_final=t_final,
        solucion_exacta=lambda t: K / (1 + ((K - y0) / y0) * np.exp(-r * t)),
        nombre="Crecimiento logístico",
        ecuacion=f"y' = {r:g}·y(1 - y/{K:g})",
    )


def oscilador_armonico(omega: float = 1.0, x0: float = 1.0, v0: float = 0.0, t_final: float = 4 * np.pi) -> ProblemaValorInicial:
    """Oscilador armónico x'' + ω²x = 0, escrito como sistema de primer orden:

        x' = v
        v' = -ω²x

    con y = [x, v]. Solución: x = x0 cos(ωt) + (v0/ω) sin(ωt).
    """

    def f(t, y):
        x, v = y
        return np.array([v, -(omega**2) * x])

    def exacta(t):
        t = np.asarray(t, dtype=float)
        x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
        v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
        return np.stack([x, v], axis=-1)

    return ProblemaValorInicial(
        f=f,
        t0=0.0,
        y0=[x0, v0],
        t_final=t_final,
        solucion_exacta=exacta,
        nombre="Oscilador armónico",
        ecuacion=f"x'' + {omega**2:g}·x = 0",
        etiquetas=("x", "v"),
    )


PROBLEMAS = {
    "exponencial": crecimiento_exponencial,
    "burden": ejemplo_burden,
    "enfriamiento": enfriamiento_newton,
    "logistica": logistica,
    "oscilador": oscilador_armonico,
}
"""Problemas disponibles por nombre (usado por la línea de comandos)."""
