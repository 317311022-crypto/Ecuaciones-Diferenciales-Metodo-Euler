"""Métodos numéricos intercambiables.

Importar este paquete registra automáticamente los métodos incluidos.
Para agregar uno nuevo, crea un módulo aquí (copia ``euler.py`` como
plantilla) e impórtalo en la lista de abajo. Aparecerá solo en la app
visual (``python -m edo.app``) y en ``resolver(..., metodo="nombre")``.
"""

from .base import Etapa, MetodoNumerico, Paso
from .registro import metodos_disponibles, obtener_metodo, registrar

# Métodos incluidos (importarlos los registra).
from .euler import Euler
from .heun import Heun
from .rk4 import RungeKutta4

__all__ = [
    "MetodoNumerico",
    "Paso",
    "Etapa",
    "registrar",
    "obtener_metodo",
    "metodos_disponibles",
    "Euler",
    "Heun",
    "RungeKutta4",
]
