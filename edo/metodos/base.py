"""Clase base para los métodos numéricos de un paso.

Todos los métodos del proyecto comparten la misma idea: a partir del punto
actual ``(t_n, y_n)`` y un tamaño de paso ``h`` calculan ``y_{n+1}``.
Lo único que cambia de un método a otro es *cómo* se da ese paso, así que
cada método solo tiene que implementar ``paso``. El ciclo que recorre el
intervalo vive en ``edo.solucionador.resolver`` y es el mismo para todos.

Traza de etapas
---------------
Para mostrar el "paso a paso" (en la app visual o en una tabla), un método
puede evaluar ``f`` con ``self.evaluar(f, t, y, "k1")`` en lugar de llamar a
``f(t, y)`` directamente. Cuando se pide ``paso_detallado``, cada evaluación
queda registrada como una :class:`Etapa` (punto donde se evaluó y pendiente
obtenida) sin cambiar el resultado numérico. Un método que llama a ``f``
directamente sigue funcionando; solo que su paso no mostrará etapas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..problema import FuncionEDO


@dataclass(frozen=True)
class Etapa:
    """Una evaluación de ``f`` dentro de un paso: ``pendiente = f(t, y)``."""

    etiqueta: str
    t: float
    y: np.ndarray
    pendiente: np.ndarray


@dataclass(frozen=True)
class Paso:
    """Registro completo de un paso ``(t_n, y_n) → (t_{n+1}, y_{n+1})``."""

    t: float
    y: np.ndarray
    h: float
    y_siguiente: np.ndarray
    etapas: tuple[Etapa, ...] = ()
    indice: int = 0

    @property
    def t_siguiente(self) -> float:
        return self.t + self.h

    @property
    def pendiente_efectiva(self) -> np.ndarray:
        """Pendiente de la recta que une ``(t_n, y_n)`` con ``(t_{n+1}, y_{n+1})``.

        En Euler coincide con ``f(t_n, y_n)``; en Heun o RK4 es un promedio
        ponderado de las pendientes de las etapas.
        """
        return (self.y_siguiente - self.y) / self.h


class MetodoNumerico(ABC):
    """Interfaz común de un método numérico de un paso.

    Para crear un método nuevo:

    1. Hereda de ``MetodoNumerico``.
    2. Define los atributos ``nombre`` (identificador único), ``nombre_completo``,
       ``orden`` (orden de convergencia global) y, si quieres, ``formula``.
    3. Implementa ``paso(f, t, y, h)``. Usa ``self.evaluar(f, t, y, "k1")`` para
       que cada evaluación de ``f`` aparezca en el registro paso a paso.
    4. Decora la clase con ``@registrar`` para poder usarla por su nombre
       (y para que aparezca en la app visual).

    Opcionalmente puedes sobrescribir ``funcion_estabilidad`` (para que la app
    avise cuando ``h`` es inestable) y ``formula_numerica`` (para mostrar la
    fórmula con los números sustituidos).
    """

    nombre: str = ""
    nombre_completo: str = ""
    orden: int = 1
    formula: str = ""

    _traza: Optional[list[Etapa]] = None

    @abstractmethod
    def paso(self, f: FuncionEDO, t: float, y: np.ndarray, h: float) -> np.ndarray:
        """Calcula ``y_{n+1}`` a partir de ``(t_n, y_n)`` con paso ``h``."""

    # ----- Instrumentación para el paso a paso -----
    def evaluar(self, f: FuncionEDO, t: float, y, etiqueta: str = "k") -> np.ndarray:
        """Evalúa ``f(t, y)`` y, si se está trazando un paso, lo registra."""
        k = np.asarray(f(t, y), dtype=float)
        if self._traza is not None:
            self._traza.append(Etapa(etiqueta, float(t), np.array(y, dtype=float), np.array(k, dtype=float)))
        return k

    def paso_detallado(self, f: FuncionEDO, t: float, y, h: float, indice: int = 0) -> Paso:
        """Igual que ``paso`` pero devuelve un :class:`Paso` con sus etapas."""
        self._traza = []
        try:
            y_sig = np.asarray(self.paso(f, t, np.asarray(y, dtype=float), h), dtype=float)
            etapas = tuple(self._traza)
        finally:
            self._traza = None
        return Paso(
            t=float(t), y=np.array(y, dtype=float), h=float(h),
            y_siguiente=y_sig, etapas=etapas, indice=indice,
        )

    # ----- Información opcional -----
    def funcion_estabilidad(self, z: complex) -> Optional[complex]:
        """Función de estabilidad ``R(z)``.

        Para la ecuación de prueba ``y' = λy`` el método da
        ``y_{n+1} = R(hλ)·y_n``. Si ``|R(hλ)| > 1`` con ``λ < 0`` el método
        amplifica lo que debería decaer: es **inestable** con ese ``h``.
        Devuelve ``None`` si el método no la define.
        """
        return None

    def formula_numerica(self, paso: Paso) -> Optional[str]:
        """Fórmula del paso con los números sustituidos (texto, puede tener
        varias líneas). ``None`` si el método no la define."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(nombre={self.nombre!r}, orden={self.orden})"

    def __str__(self) -> str:
        return self.nombre_completo or self.nombre
