"""El dibujo por capas debe verse EXACTAMENTE igual que un dibujo completo.

La ventana solo copia a la pantalla las zonas que cambian (``blit``). Si una
capa olvida repintarse o copiar su zona, en pantalla queda basura vieja. Aquí
se simula una pantalla como la de Tk: ``draw()`` copia todo y ``blit(zona)``
copia solo esa zona. Después de cada interacción, la "pantalla" se compara
píxel por píxel contra un dibujo completo del mismo estado.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseEvent, ResizeEvent  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402

from edo import Configuracion  # noqa: E402
from edo.app import EulerLab  # noqa: E402


class LienzoConPantalla(FigureCanvasAgg):
    """Como TkAgg: ``draw`` pinta todo y lo muestra; ``blit(zona)`` muestra solo esa zona."""

    def __init__(self, figure):
        super().__init__(figure)
        self.pantalla = None

    def draw(self, *args, **kwargs):
        super().draw(*args, **kwargs)
        self.pantalla = np.asarray(self.buffer_rgba()).copy()

    def blit(self, bbox=None):
        buf = np.asarray(self.buffer_rgba())
        if self.pantalla is None or self.pantalla.shape != buf.shape or bbox is None:
            self.pantalla = buf.copy()
            return
        alto, ancho = buf.shape[:2]
        x0, y0, x1, y1 = bbox.extents
        x0, x1 = max(int(np.floor(x0)), 0), min(int(np.ceil(x1)), ancho)
        y0, y1 = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)), alto)
        self.pantalla[alto - y1:alto - y0, x0:x1] = buf[alto - y1:alto - y0, x0:x1]


def _referencia(app):
    """Dibujo completo del estado actual de los artistas, sin capas."""
    for a in app._animados:
        a.set_animated(False)
    app._guardando = True
    try:
        app.consola.renderizar()
        FigureCanvasAgg.draw(app.fig.canvas)
        return np.asarray(app.fig.canvas.buffer_rgba()).copy()
    finally:
        for a in app._animados:
            a.set_animated(True)
        app._guardando = False


def _anillos_de_widgets(app, alto, ancho):
    """Bordes (±2 px) de botones y campos: al repintar un widget suelto, matplotlib mezcla
    otra vez su borde antialiasado (pasa igual con el resaltado de sus botones). Es
    invisible y no es parte de las capas; el interior del widget sí se compara."""
    mascara = np.zeros((alto, ancho), dtype=bool)
    widgets = [b.ax for b in app.botones.values()] + [app.boton_metodo.ax, app.boton_resolver.ax]
    widgets += [tb.ax for tb in app.cajas.values()]
    for ax in widgets:
        x0, y0, x1, y1 = ax.bbox.extents
        for xa, xb, ya, yb in ((x0 - 2, x0 + 2, y0 - 2, y1 + 2), (x1 - 2, x1 + 2, y0 - 2, y1 + 2),
                               (x0 - 2, x1 + 2, y0 - 2, y0 + 2), (x0 - 2, x1 + 2, y1 - 2, y1 + 2)):
            c0, c1 = max(int(np.floor(xa)), 0), min(int(np.ceil(xb)), ancho)
            f0, f1 = max(alto - int(np.ceil(yb)), 0), min(alto - int(np.floor(ya)), alto)
            mascara[f0:f1, c0:c1] = True
    return mascara


def _comparar(app, que):
    pantalla = app.fig.canvas.pantalla.copy()
    ref = _referencia(app)
    assert pantalla.shape == ref.shape, que
    distintos = (np.abs(pantalla.astype(int) - ref.astype(int)).max(axis=2) > 8)
    distintos &= ~_anillos_de_widgets(app, *distintos.shape)
    if distintos.any():
        ys, xs = np.nonzero(distintos)
        pytest.fail(f"{que}: {distintos.sum()} píxeles distintos en x∈[{xs.min()}, {xs.max()}], "
                    f"y∈[{ys.min()}, {ys.max()}] (desde arriba)")


def _cuadros(app, n, dt=0.06):
    for _ in range(n):
        app._t_prev -= dt
        app._tick()


def _raton(app, ax, fx, fy, tipo="motion_notify_event", **kw):
    x, y = ax.transAxes.transform((fx, fy))
    MouseEvent(tipo, app.fig.canvas, x, y, **kw)._process()


@pytest.mark.parametrize("calidad", ["media", "alta", "baja"])
def test_pantalla_igual_a_dibujo_completo(calidad):
    app = EulerLab(Configuracion("x + y", 0, 1, 0.1, 1.5, "euler", "2exp(x) - x - 1"), calidad=calidad,
                   figsize=(15.5, 8.2), dpi=80, animar=True)
    c = LienzoConPantalla(app.fig)
    app._invalidar()
    c.draw()
    _comparar(app, "inicio")

    _cuadros(app, 3)
    _comparar(app, "cuadros dentro del primer paso")
    _cuadros(app, 25)
    _comparar(app, "cuadros revelando pasos")

    _raton(app, app.grafica.ax, 0.4, 0.5)
    _cuadros(app, 2)
    _comparar(app, "ratón sobre la gráfica")
    _raton(app, app.grafica.ax, 0.6, 0.3)
    _comparar(app, "ratón: solo la barra de estado")

    _cuadros(app, 200)                       # termina la animación
    assert not app.animando
    _comparar(app, "fin de la animación")

    app.ir_a(10, 2)
    i = next(i for i, ln in enumerate(app.consola.lineas) if ln.paso == 6)
    app.consola.inicio = i - 1
    app.refrescar()
    renglon = i - app.consola.inicio            # (el registro pudo ajustar el desplazamiento)
    _raton(app, app.consola.ax, 0.5, 1 - app.consola.margen_sup - (renglon + 0.5) * app.consola.fila,
           "button_press_event", button=1)
    assert app.seleccion == 6
    _comparar(app, "clic en el registro")

    _raton(app, app.consola.ax, 0.5, 0.5, "scroll_event", step=-2)
    _comparar(app, "rueda en el registro")

    _raton(app, app.grafica.ax, 0.5, 0.5, "scroll_event", step=1)
    _cuadros(app, 1)
    _comparar(app, "zoom")

    _raton(app, app.grafica.ax, 0.5, 0.5, "button_press_event", button=1)
    _raton(app, app.grafica.ax, 0.3, 0.6)
    _cuadros(app, 2)
    _raton(app, app.grafica.ax, 0.3, 0.6, "button_release_event", button=1)
    _comparar(app, "arrastrar la vista")

    app.alternar_seguir()
    app.alternar_animacion()
    _cuadros(app, 12)
    _comparar(app, "cámara 'seguir' animando")
    app.alternar_animacion()
    app.alternar_seguir()
    _cuadros(app, 120)
    _comparar(app, "regreso a la vista inicial")

    # Escribir en un campo: matplotlib redibuja la ventana completa en cada tecla.
    caja = app.cajas["h"]
    _raton(app, caja.ax, 0.5, 0.5, "button_press_event", button=1)
    _raton(app, caja.ax, 0.5, 0.5, "button_release_event", button=1)
    caja.set_val("")
    for tecla in "0.25":
        KeyEvent("key_press_event", c, tecla)._process()
    _comparar(app, "escribir en un campo")
    KeyEvent("key_press_event", c, "enter")._process()
    _cuadros(app, 8)
    _comparar(app, "resolver con h nuevo y animar")

    _raton(app, app.botones["siguiente"].ax, 0.5, 0.5)       # resaltado del botón (blit propio)
    _comparar(app, "ratón sobre un botón")

    app.siguiente_metodo()
    _cuadros(app, 5)
    _comparar(app, "cambio de método")

    app.cajas["ecuacion"].set_val("x + sin y")
    app.resolver()
    _comparar(app, "error en la ecuación")

    app.fig.set_size_inches(14, 7.6)
    ResizeEvent("resize_event", c)._process()
    _comparar(app, "cambio de tamaño de la ventana")

    import matplotlib.pyplot as plt

    plt.close(app.fig)


def test_ajuste_a_pantallas_de_laptop():
    from edo.app.ventana import DISENO, tamano_para_pantalla

    for ancho, alto in ((1536, 864), (1366, 768), (1280, 720), (1920, 1080), (2560, 1440)):
        (w, h), dpi = tamano_para_pantalla((ancho, alto))
        assert w * dpi <= ancho - 20, (ancho, alto)              # cabe a lo ancho
        assert h * dpi <= max(alto - 100, DISENO[0] / 2.15 * dpi), (ancho, alto)   # y a lo alto
        assert 1.6 <= w / h <= 2.16                              # el diseño no se deforma demasiado
        assert dpi <= 100


def test_perfiles_de_calidad():
    from edo.app.calidad import CALIDADES, obtener_calidad

    assert obtener_calidad("MEDIA") is CALIDADES["media"]
    assert CALIDADES["media"].brillo < CALIDADES["alta"].brillo and not CALIDADES["baja"].decoracion
    for q in CALIDADES.values():
        n, a = q.halo(9, 0.045)
        assert 1 <= n <= 9 and n * a == pytest.approx(9 * 0.045)       # el brillo total se conserva
        assert 0 < q.intervalo_ms <= 50
    with pytest.raises(ValueError):
        obtener_calidad("ultra")
