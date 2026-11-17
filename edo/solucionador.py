"""Ciclo de integración común a todos los métodos y objeto ``Solucion``.

``resolver`` no sabe nada de Euler ni de RK4: recibe un método (por nombre o
instancia) y llama a ``metodo.paso`` en cada punto de la malla. Por eso,
cambiar de método no requiere tocar este archivo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metodos import MetodoNumerico, obtener_metodo
from .problema import ProblemaValorInicial


@dataclass
class Solucion:
    """Resultado de integrar un PVI con un método numérico.

    Atributos
    ---------
    t : ndarray, forma (n+1,)
        Nodos de la malla ``t_0, t_1, ..., t_n``.
    y : ndarray, forma (n+1,) o (n+1, m)
        Aproximaciones ``y_0, y_1, ..., y_n``.
    """

    t: np.ndarray
    y: np.ndarray
    metodo: MetodoNumerico
    h: float
    problema: ProblemaValorInicial

    @property
    def n_pasos(self) -> int:
        return len(self.t) - 1

    @property
    def nombre_metodo(self) -> str:
        return str(self.metodo)

    # ----- Errores (requieren solución exacta) -----
    def exacta(self) -> np.ndarray:
        """Solución exacta evaluada en los nodos de la malla."""
        return self.problema.exacta(self.t)

    def error_absoluto(self) -> np.ndarray:
        """|y(t_i) - y_i| en cada nodo (norma infinito si es sistema)."""
        dif = np.abs(self.exacta() - self.y)
        return dif.max(axis=1) if dif.ndim == 2 else dif

    def error_relativo(self) -> np.ndarray:
        exacta = self.exacta()
        norma = np.abs(exacta).max(axis=1) if exacta.ndim == 2 else np.abs(exacta)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(norma > 0, self.error_absoluto() / norma, np.nan)

    def error_maximo(self) -> float:
        """Máximo error absoluto sobre toda la malla."""
        return float(self.error_absoluto().max())

    def error_final(self) -> float:
        """Error absoluto en ``t_final``."""
        return float(self.error_absoluto()[-1])

    # ----- Presentación -----
    def tabla(self, cada: int = 1, decimales: int = 6) -> str:
        """Tabla de texto con t_i, y_i y, si existe, la exacta y el error."""
        con_exacta = self.problema.tiene_solucion_exacta and not self.problema.es_sistema
        ancho = decimales + 8
        encabezado = f"{'i':>4} {'t_i':>{ancho}} {'y_i':>{ancho}}"
        if con_exacta:
            encabezado += f" {'y(t_i)':>{ancho}} {'|error|':>{ancho}}"
        lineas = [encabezado, "-" * len(encabezado)]

        exacta = self.exacta() if con_exacta else None
        indices = list(range(0, len(self.t), max(1, cada)))
        if indices[-1] != len(self.t) - 1:
            indices.append(len(self.t) - 1)

        for i in indices:
            if self.problema.es_sistema:
                yi = "[" + ", ".join(f"{v:.{decimales}f}" for v in self.y[i]) + "]"
                lineas.append(f"{i:>4} {self.t[i]:>{ancho}.{decimales}f} {yi}")
                continue
            fila = f"{i:>4} {self.t[i]:>{ancho}.{decimales}f} {self.y[i]:>{ancho}.{decimales}f}"
            if exacta is not None:
                err = abs(exacta[i] - self.y[i])
                fila += f" {exacta[i]:>{ancho}.{decimales}f} {err:>{ancho}.{decimales}e}"
            lineas.append(fila)
        return "\n".join(lineas)

    def __repr__(self) -> str:
        return (
            f"Solucion(metodo={self.metodo.nombre!r}, h={self.h:g}, "
            f"n_pasos={self.n_pasos}, problema={self.problema.nombre!r})"
        )


def _construir_malla(t0: float, t_final: float, h: Optional[float], n: Optional[int]) -> tuple[np.ndarray, float]:
    """Construye los nodos ``t_i``. Si ``h`` no divide exactamente al
    intervalo, el último paso se acorta para terminar justo en ``t_final``."""
    if (h is None) == (n is None):
        raise ValueError("Indica exactamente uno de 'h' (tamaño de paso) o 'n' (número de pasos).")
    longitud = t_final - t0
    if n is not None:
        if n < 1:
            raise ValueError("'n' debe ser un entero positivo.")
        return np.linspace(t0, t_final, int(n) + 1), longitud / n
    if h <= 0:
        raise ValueError("'h' debe ser positivo.")
    cociente = longitud / h
    n_completos = round(cociente)
    if math.isclose(cociente, n_completos, rel_tol=1e-9, abs_tol=1e-9):
        return np.linspace(t0, t_final, n_completos + 1), h
    n_completos = math.floor(cociente)
    t = t0 + h * np.arange(n_completos + 1)
    return np.append(t, t_final), h


def resolver(
    problema: ProblemaValorInicial,
    metodo: str | MetodoNumerico = "euler",
    h: Optional[float] = None,
    n: Optional[int] = None,
) -> Solucion:
    """Resuelve numéricamente un PVI.

    Parámetros
    ----------
    problema : ProblemaValorInicial
    metodo : str o MetodoNumerico
        Nombre de un método registrado (``"euler"``, ``"heun"``, ``"rk4"``)
        o una instancia. Por defecto, Euler.
    h : float, opcional
        Tamaño de paso.
    n : int, opcional
        Número de pasos (alternativa a ``h``).

    Ejemplo
    -------
    >>> from edo import resolver, ejemplos
    >>> sol = resolver(ejemplos.crecimiento_exponencial(), "euler", h=0.1)
    """
    met = obtener_metodo(metodo)
    t, h_nominal = _construir_malla(problema.t0, problema.t_final, h, n)

    y = np.empty((len(t),) + problema.y0.shape, dtype=float)
    y[0] = problema.y0
    for i in range(len(t) - 1):
        h_i = t[i + 1] - t[i]
        y[i + 1] = met.paso(problema.f, t[i], y[i], h_i)

    return Solucion(t=t, y=y, metodo=met, h=h_nominal, problema=problema)


def orden_observado(problema: ProblemaValorInicial, metodo: str | MetodoNumerico, pasos) -> list[tuple[float, float, float]]:
    """Estima el orden de convergencia: resuelve con varios ``h`` y compara
    errores consecutivos con ``p ≈ log(e1/e2) / log(h1/h2)``.

    Devuelve una lista de tuplas ``(h, error_maximo, p)``; ``p`` es ``nan``
    para el primer renglón.
    """
    resultados = []
    anterior = None
    for h in pasos:
        err = resolver(problema, metodo, h=h).error_maximo()
        p = math.nan
        if anterior is not None and err > 0 and anterior[1] > 0:
            p = math.log(anterior[1] / err) / math.log(anterior[0] / h)
        resultados.append((h, err, p))
        anterior = (h, err)
    return resultados
