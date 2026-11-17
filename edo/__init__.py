"""edo — Solución numérica de ecuaciones diferenciales ordinarias.

Uso básico::

    from edo import ProblemaValorInicial, resolver

    pvi = ProblemaValorInicial(f=lambda t, y: y - t**2 + 1, t0=0, y0=0.5, t_final=2)
    sol = resolver(pvi, metodo="euler", h=0.2)
    print(sol.tabla())

Para cambiar de método basta con cambiar el nombre: ``metodo="rk4"``.

App visual (registro paso a paso + gráfica)::

    python -m edo.app            # o:  python euler_app.py
"""

from . import ejemplos
from .expresion import ErrorExpresion, compilar_edo, compilar_funcion
from .metodos import Etapa, MetodoNumerico, Paso, metodos_disponibles, obtener_metodo, registrar
from .problema import ProblemaValorInicial
from .simulacion import Aviso, Configuracion, Simulacion
from .solucionador import Solucion, orden_observado, resolver

__version__ = "0.2.0"

__all__ = [
    "ProblemaValorInicial",
    "Solucion",
    "resolver",
    "orden_observado",
    "MetodoNumerico",
    "Paso",
    "Etapa",
    "registrar",
    "obtener_metodo",
    "metodos_disponibles",
    "Configuracion",
    "Simulacion",
    "Aviso",
    "compilar_edo",
    "compilar_funcion",
    "ErrorExpresion",
    "ejemplos",
]
