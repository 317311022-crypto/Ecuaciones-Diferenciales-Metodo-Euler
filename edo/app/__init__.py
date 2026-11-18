"""App visual del método de Euler (y de cualquier método registrado).

Uso::

    python -m edo.app                       # valores por defecto
    python euler_app.py                     # con la configuración escrita al inicio del archivo
    python -m edo.app --calidad alta        # todos los efectos (parallax); "media" para laptops

    from edo.app import EulerLab
    from edo import Configuracion
    EulerLab(Configuracion("x + y", x0=0, y0=1, h=0.1, x_final=2)).mostrar()

Módulos:

* ``tema``     colores y fuentes (cámbialos aquí).
* ``calidad``  perfiles alta / media / baja (cuántos efectos dibujar).
* ``efectos``  luminiscencia y fondo con parallax.
* ``consola``  el registro paso a paso con resaltado de sintaxis.
* ``grafica``  la gráfica: campo de pendientes, poligonal, tangente, etapas, error.
* ``ventana``  arma todo y maneja eventos y animación.
"""

__all__ = ["EulerLab", "main"]


def __getattr__(nombre):
    # Importación diferida: ``from edo.app import tema`` no carga la ventana completa.
    if nombre in __all__:
        from . import ventana

        return getattr(ventana, nombre)
    raise AttributeError(f"module 'edo.app' has no attribute {nombre!r}")
