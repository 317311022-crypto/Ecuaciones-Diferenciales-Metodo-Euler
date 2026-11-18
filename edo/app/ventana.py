"""Ventana principal: barra de entradas arriba, registro a la izquierda y
gráfica a la derecha.

Todo está hecho con matplotlib (sin dependencias extra): los campos de texto
y botones son ``matplotlib.widgets``, el registro es :class:`Consola` y la
gráfica :class:`PanelGrafica`. Un temporizador (20-30 fps según la calidad)
anima el paso a paso y la cámara, y se apaga solo cuando nada se mueve.

Rendimiento: dibujo por capas
-----------------------------
Matplotlib dibuja con el CPU, y redibujar la ventana completa cuesta mucho.
Por eso la ventana se compone de capas que se guardan ya dibujadas
(``copy_from_bbox``) y solo se repintan cuando cambia lo que muestran:

* **fija**: barras, paneles, botones, fondo de la gráfica. Solo se redibuja al
  escribir en un campo, cambiar de método o de tamaño de ventana.
* **registro**: el panel izquierdo. Al revelar o seleccionar un paso.
* **base**: ejes, campo de pendientes, referencia y franja de error. Al mover
  la vista (zoom, arrastrar, cámara "seguir") o cambiar de paso.
* **dinámica**: la poligonal y la geometría del paso. En cada cuadro.
* **estado** y **cursor**: textos de la barra inferior y cursor del registro.

Cada actualización restaura la zona guardada de lo que no cambió, dibuja
encima solo lo nuevo y copia a la pantalla (``blit``) solo esa zona.

Atajos (con el foco fuera de los campos de texto)::

    espacio  animar / pausar          ← →   paso anterior / siguiente
    inicio   volver al paso 0          fin   mostrar todos los pasos
    c        cámara que sigue al punto  r    restablecer la vista
    + / -    velocidad                  m    siguiente método
    rueda    zoom (gráfica) o desplazar (registro); arrastrar = mover la vista
"""

from __future__ import annotations

import math
import platform
import sys
import textwrap
import time
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.colors import same_color, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, TextBox

from ..expresion import ErrorExpresion
from ..formato import corto, etiqueta, num, sub
from ..metodos import metodos_disponibles, obtener_metodo
from ..simulacion import Configuracion, Simulacion
from .calidad import Calidad, obtener_calidad
from .consola import Consola, L, Linea, colorear
from .efectos import FondoParallax
from .grafica import PanelGrafica
from .tema import CONSOLA, ETAPAS, FUENTE_MONO, FUENTE_UI, GRAFICA, UI, aplicar_tema

# ---------------------------------------------------------------- geometría (fracción de figura)
TITULO_Y = 0.958
ENTRADAS_Y = 0.885
CUERPO_ARRIBA = 0.872
ESTADO_ALTO = 0.032
CABECERA = 0.038
IZQ = (0.006, 0.358)                     # x inicial y ancho del panel de registro
DER = (0.370, 0.624)                     # x inicial y ancho del panel de gráfica
PAUSA_EN_NODO = 0.35                     # fracción de paso que se queda quieto al llegar a un nodo
DISENO = (15.5, 8.7)                     # pulgadas de la figura; a 100 dpi son 1550 x 870 píxeles lógicos
Y_LEYENDA = 0.268                        # renglón de la leyenda, entre la gráfica y la franja de error


def _pantalla_logica() -> Optional[tuple[float, float]]:
    """Tamaño de la pantalla en píxeles *lógicos* (los de Windows con su escala,
    p. ej. 1536 x 864 para 1920 x 1080 al 125 %), o ``None`` si no se puede saber.
    Solo con el backend Tk, que es el predeterminado de Python en Windows."""
    try:
        if "tk" not in matplotlib.get_backend().lower():
            return None
        import tkinter

        raiz = tkinter.Tk()
    except Exception:
        return None
    try:
        raiz.withdraw()
        ancho, alto = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
        escala = max(raiz.winfo_fpixels("1i") / 96.0, 1.0)   # > 1 si Tk ya trabaja en píxeles físicos
    except Exception:
        return None
    finally:
        raiz.destroy()
    return ancho / escala, alto / escala


def tamano_para_pantalla(pantalla: tuple[float, float]) -> tuple[tuple[float, float], float]:
    """``(figsize, dpi)`` para que la ventana quepa en una pantalla de ``pantalla``
    = (ancho, alto) píxeles lógicos, dejando lugar a la barra de tareas.

    El ancho se ajusta bajando los dpi (así letras y márgenes se encogen juntos y
    el diseño no se deforma); el alto se aprovecha completo."""
    ancho, alto = pantalla
    escala = float(np.clip((ancho - 24) / (DISENO[0] * 100), 0.6, 1.0))
    dpi = 100 * escala
    alto_in = float(np.clip((alto - 110) / dpi, DISENO[0] / 2.15, DISENO[1] * 1.08))
    return (DISENO[0], alto_in), dpi


def _colocar_ventana(fig) -> None:
    """Ventana arriba a la izquierda y, en Windows, maximizada (backend Tk)."""
    try:
        ventana = fig.canvas.manager.window
        if not hasattr(ventana, "wm_geometry"):
            return
        ventana.wm_geometry("+0+0")
        if sys.platform.startswith("win"):
            # Después de que matplotlib la muestre (show() la "desiconifica" en estado normal).
            ventana.after(120, lambda: ventana.state("zoomed"))
    except Exception:
        pass


class CampoTexto(TextBox):
    """``TextBox`` que, al escribir o al pasar el ratón, repinta solo su rectángulo.

    El ``TextBox`` de matplotlib redibuja la ventana completa en cada tecla, que
    es lo más caro que hay. ``repintar`` (lo asigna la ventana) dibuja solo este
    campo y lo copia a la pantalla; si todavía no se puede, se usa el
    comportamiento normal."""

    repintar = None

    def _puede_repintar(self) -> bool:
        return self.repintar is not None and self.repintar(self, solo_probar=True)

    def _rendercursor(self):
        if not self._puede_repintar():
            return super()._rendercursor()
        # Mismo cálculo que matplotlib para colocar el cursor, sin redibujar todo.
        texto = self.text_disp.get_text()
        hasta_cursor = texto[: self.cursor_index]
        caja = self.text_disp.get_window_extent()
        self.text_disp.set_text(hasta_cursor or ",")
        caja_ancho = self.text_disp.get_window_extent()
        if caja.y0 == caja.y1:
            caja.y0 -= caja_ancho.height / 2
            caja.y1 += caja_ancho.height / 2
        elif not hasta_cursor:
            caja.x1 = caja.x0
        else:
            caja.x1 = caja.x0 + caja_ancho.width
        self.cursor.set(segments=[[(caja.x1, caja.y0), (caja.x1, caja.y1)]], visible=True)
        self.text_disp.set_text(texto)
        self.repintar(self)

    def _motion(self, event):
        if self.ignore(event):
            return
        color = self.hovercolor if self.ax.contains(event)[0] else self.color
        if not same_color(color, self.ax.get_facecolor()):
            self.ax.set_facecolor(color)
            if self.drawon and not (self._puede_repintar() and self.repintar(self)):
                self.ax.figure.canvas.draw()


def _caja_px(x0, y0, x1, y1, pad, ancho, alto) -> Bbox:
    """Rectángulo en píxeles enteros, ampliado ``pad`` y recortado a la figura."""
    return Bbox.from_extents(max(0, math.floor(x0) - pad), max(0, math.floor(y0) - pad),
                             min(ancho, math.ceil(x1) + pad), min(alto, math.ceil(y1) + pad))


def _orden_metodos() -> list[str]:
    nombres = metodos_disponibles()
    return sorted(nombres, key=lambda n: (n != "euler", n))


class EulerLab:
    """La aplicación. ``EulerLab(Configuracion(...), calidad="media").mostrar()``.

    ``calidad``: ``"alta"`` (con parallax), ``"media"`` (laptops) o ``"baja"``;
    ver :mod:`edo.app.calidad`. ``figsize``/``dpi``: si se omiten, se calculan
    para que la ventana quepa en la pantalla (``ajustar_pantalla=True``).
    """

    def __init__(self, config: Optional[Configuracion] = None, calidad: "str | Calidad" = "media",
                 figsize=None, dpi: Optional[float] = None, animar: bool = True, ajustar_pantalla: bool = True):
        aplicar_tema(app=True)
        import matplotlib.pyplot as plt

        self.calidad = obtener_calidad(calidad)
        if figsize is None or dpi is None:
            pantalla = _pantalla_logica() if ajustar_pantalla else None
            tam, d = tamano_para_pantalla(pantalla) if pantalla else (DISENO, 100.0)
            figsize = figsize or tam
            dpi = dpi or d
        self.config = config or Configuracion()
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=UI["fondo"])
        try:
            self.fig.canvas.manager.set_window_title("Euler·Lab — método de Euler paso a paso")
        except AttributeError:
            pass
        if ajustar_pantalla:
            _colocar_ventana(self.fig)

        self.sim: Optional[Simulacion] = None
        self.revelados = 0
        self.seleccion = 0
        self.progreso = 1.0
        self.animando = False
        self.velocidad = 2.5
        self.seguir = False
        self.metodos = _orden_metodos()
        self.metodo = self.config.metodo if self.config.metodo in self.metodos else self.metodos[0]
        self.raton = np.zeros(2)
        self.raton_suave = np.zeros(2)
        self.camara = np.zeros(2)
        self._objetivo_vista = None
        self._arrastre = None
        self._t0 = time.perf_counter()
        self._t_prev = self._t0
        self._timer = None
        self._timer_activo = False
        self._error_txt: Optional[str] = None
        self._bloques: list[list[Linea]] = []
        self._encabezado: list[Linea] = []
        self._final: list[Linea] = []
        self._lineas_error: list[Linea] = []
        self._version_registro = 0
        self._clave_lineas = None
        self._lineas_cache: list[Linea] = []
        self._pulso = 1.0

        q = self.calidad
        self._cromo()
        self.fondo = FondoParallax(self.fig, self._rect_grafica_panel(), parallax=q.parallax,
                                   decoracion=q.decoracion, centelleo=q.centelleo, zorder=-5)
        x, w = DER
        self.grafica = PanelGrafica(self.fig, [x + 0.040, 0.318, w - 0.052, 0.500],
                                    [x + 0.040, 0.075, w - 0.052, 0.155], calidad=q)
        x, w = IZQ
        self.consola = Consola(self.fig, [x + 0.002, ESTADO_ALTO + 0.010, w - 0.004,
                                          CUERPO_ARRIBA - CABECERA - ESTADO_ALTO - 0.012])
        self._entradas()
        self._controles()
        self._barra_estado()

        # Capas (ver el docstring del módulo). Lo marcado como "animated" queda fuera del
        # dibujo completo de matplotlib y lo pinta este módulo, cada capa en su zona.
        self.cursor = self.consola.ax.text(0, 0, "▌", color=CONSOLA["titulo"], family=FUENTE_MONO,
                                           fontsize=self.consola.tam, va="baseline", visible=False, zorder=3)
        g = self.grafica
        self._capa_registro = [self.consola.ax, self.txt_cabecera_registro]
        # Bases de la gráfica: nombre -> (zona, artistas, sub-zonas de las capas que van encima).
        if self.fondo.parallax:
            # El fondo se mueve: cada vez hay que redibujar todo el panel (y la leyenda encima).
            self._capa_fondo = [self.fondo.ax_fijo]
            self._bases = {"base": ("grafica", [self.fondo.ax, g.ax, g.ax_err, *self.leyenda],
                                    ("principal", "franja"))}
        else:
            # Fondo fijo: imagen, decoración y leyenda se dibujan una vez por tamaño de ventana;
            # los ejes de arriba (siguen la vista) y los de la franja (vista general) van aparte.
            self._capa_fondo = [self.fondo.ax_fijo, self.fondo.ax, *self.leyenda]
            self._bases = {"base": ("sup", [g.ax], ("principal",)),
                           "base_err": ("inf", [g.ax_err], ("franja",))}
        self._capa_franja = [g.capa_err]
        self._capa_dinamica = [g.capa]
        self._capa_estado = [self.est_estado, self.est_metodo, self.est_param, self.est_error, self.est_coord]
        self._animados = [*self._capa_registro, *self._capa_fondo,
                          *[a for _, artistas, _ in self._bases.values() for a in artistas],
                          *self._capa_franja, *self._capa_dinamica, *self._capa_estado, self.cursor]
        self._animados = list(dict.fromkeys(self._animados))
        for a in self._animados:
            a.set_animated(True)
        self._est: Optional[dict] = None     # zonas de la capa fija tras el último dibujo completo
        self._cache: dict = {}               # zonas ya dibujadas de registro / base / zona principal
        self._firmas: dict = {}              # qué muestra ahora cada capa
        self._tam = None                     # tamaño de la figura con el que se llenaron los cachés
        self._regiones_cache = None
        self._fotos_campos: dict = {}
        self._guardando = False
        self._sucio = False

        c = self.fig.canvas
        c.mpl_connect("draw_event", self._al_dibujar)
        c.mpl_connect("scroll_event", self._al_rodar)
        c.mpl_connect("button_press_event", self._al_presionar)
        c.mpl_connect("button_release_event", self._al_soltar)
        c.mpl_connect("motion_notify_event", self._al_mover)
        c.mpl_connect("key_press_event", self._al_teclear)
        c.mpl_connect("resize_event", self._al_redimensionar)
        c.mpl_connect("figure_leave_event", self._al_salir)
        c.mpl_connect("close_event", lambda _e: self._dormir())

        self.resolver(animar=animar)

    # =================================================================== construcción
    def _rect_grafica_panel(self):
        x, w = DER
        return [x, ESTADO_ALTO + 0.008, w, CUERPO_ARRIBA - CABECERA - ESTADO_ALTO - 0.008]

    def _cromo(self) -> None:
        """Fondos, bordes y textos fijos de la ventana (un eje que cubre toda la figura)."""
        ax = self.cromo = self.fig.add_axes([0, 0, 1, 1], zorder=-10)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_navigate(False)
        R = lambda *a, **k: ax.add_patch(Rectangle(*a, **k))  # noqa: E731

        # Barra de título con "semáforo".
        R((0, TITULO_Y), 1, 1 - TITULO_Y, color=UI["panel_alto"], lw=0)
        R((0, TITULO_Y), 1, 0.0015, color=UI["borde"], lw=0)
        ax.scatter([0.011, 0.0225, 0.034], [(1 + TITULO_Y) / 2] * 3, s=52,
                   color=["#ff5f57", "#febc2e", "#28c840"], linewidths=0)
        ax.text(0.052, (1 + TITULO_Y) / 2, "◆", color=UI["acento"], fontsize=11, va="center",
                path_effects=_brillo(UI["acento"]))
        ax.text(0.064, (1 + TITULO_Y) / 2, "EULER·LAB", color=UI["texto"], fontsize=10.5, va="center",
                fontweight="bold", family=FUENTE_UI)
        ax.text(0.128, (1 + TITULO_Y) / 2, "método de Euler · ecuaciones diferenciales ordinarias",
                color=UI["texto_tenue"], fontsize=9, va="center")
        ax.text(0.994, (1 + TITULO_Y) / 2,
                f"Python {platform.python_version()}  ·  matplotlib {matplotlib.__version__}",
                color=UI["texto_tenue"], fontsize=8, va="center", ha="right", family=FUENTE_MONO)

        # Barra de entradas.
        R((0, ENTRADAS_Y), 1, TITULO_Y - ENTRADAS_Y, color=UI["panel"], lw=0)
        R((0, ENTRADAS_Y), 1, 0.0015, color=UI["borde"], lw=0)

        # Paneles del cuerpo.
        for (x, w), titulo, icono, color in ((IZQ, "REGISTRO", "▸", UI["acento_2"]),
                                              (DER, "VISUAL", "◈", UI["acento"])):
            y0 = ESTADO_ALTO + 0.006
            alto = CUERPO_ARRIBA - y0
            R((x, y0), w, alto, facecolor=UI["panel"], edgecolor=UI["borde"], lw=1.0)
            R((x, CUERPO_ARRIBA - CABECERA), w, CABECERA, facecolor=UI["panel_alto"], edgecolor=UI["borde"], lw=1.0)
            # pestaña activa
            R((x + 0.004, CUERPO_ARRIBA - CABECERA), 0.092, CABECERA - 0.002, facecolor=UI["panel"], lw=0)
            R((x + 0.004, CUERPO_ARRIBA - 0.0025), 0.092, 0.0022, color=color, lw=0)
            yc = CUERPO_ARRIBA - CABECERA / 2
            ax.text(x + 0.010, yc, icono, color=color, fontsize=10, va="center", path_effects=_brillo(color))
            ax.text(x + 0.022, yc, titulo, color=UI["texto"], fontsize=8.5, va="center", fontweight="bold")
        self.txt_cabecera_registro = ax.text(IZQ[0] + IZQ[1] - 0.008, CUERPO_ARRIBA - CABECERA / 2, "",
                                             color=UI["texto_tenue"], fontsize=8, va="center", ha="right",
                                             family=FUENTE_MONO)
        ax.text(IZQ[0] + 0.104, CUERPO_ARRIBA - CABECERA / 2, "paso_a_paso.log", color=UI["texto_tenue"],
                fontsize=8, va="center", family=FUENTE_MONO)

        # Leyenda (marca de color + texto neutro), entre la gráfica y la franja de error.
        tokens = []
        for marca, texto, color in (("━━", "solución numérica", GRAFICA["metodo"]),
                                    ("━━", "referencia", GRAFICA["referencia"]),
                                    ("╌╌", "tangente / paso", GRAFICA["tangente"]),
                                    ("●", "k₁ k₂… (etapas)", ETAPAS[1]),
                                    ("┃", "error", GRAFICA["error"]),
                                    ("╱", "campo de pendientes", GRAFICA["campo_alto"])):
            tokens += [(marca + " ", color), (texto + "    ", UI["texto_medio"])]
        # (en la figura, no en este eje: aquí quedaría tapada por el fondo de la gráfica)
        self.leyenda = texto_coloreado(self.fig, DER[0] + 0.040, Y_LEYENDA, tokens, fontsize=7.8, va="center",
                                       zorder=5)
        self.formula_textos = []

    def _entradas(self) -> None:
        """Campos de la parte superior: ecuación, condición inicial, paso, intervalo..."""
        c = self.config
        y, alto = ENTRADAS_Y + 0.010, 0.040
        cap = ENTRADAS_Y + 0.060
        self.cajas: dict[str, CampoTexto] = {}
        especificacion = [
            # clave, etiqueta, x, ancho, valor, título del grupo
            ("ecuacion", "y′ =", 0.040, 0.215, c.ecuacion, ("ECUACIÓN  y′ = f(x, y)", 0.012)),
            ("x0", "x₀", 0.290, 0.048, corto_o_texto(c.x0), ("CONDICIÓN INICIAL", 0.270)),
            ("y0", "y₀", 0.364, 0.052, corto_o_texto(c.y0), None),
            ("h", "h", 0.440, 0.052, corto_o_texto(c.h), ("PASO", 0.428)),
            ("x_final", "hasta x =", 0.556, 0.052, corto_o_texto(c.x_final), ("INTERVALO", 0.512)),
            ("exacta", "y(x) =", 0.660, 0.150, c.exacta, ("SOLUCIÓN EXACTA (opcional)", 0.626)),
        ]
        for clave, etiq, x, ancho, valor, grupo in especificacion:
            ax = self.fig.add_axes([x, y, ancho, alto], zorder=3)
            ancho_px = ancho * self.fig.get_figwidth() * self.fig.dpi
            tb = CampoTexto(ax, etiq + " ", initial=str(valor), color=UI["campo"], hovercolor=UI["campo_hover"],
                            label_pad=6 / ancho_px)
            tb.repintar = self._repintar_campo
            tb.text_disp.set_clip_on(True)       # el texto largo no se sale del campo
            tb.text_disp.set_color(UI["texto"])
            tb.text_disp.set_fontfamily(FUENTE_MONO)
            tb.text_disp.set_fontsize(10)
            tb.label.set_color(CONSOLA["variable"])
            tb.label.set_fontfamily(FUENTE_MONO)
            tb.label.set_fontsize(9.5)
            tb.cursor.set_color(UI["acento_2"])
            for s in ax.spines.values():
                s.set_color(UI["borde"])
            tb.on_submit(lambda _texto: self.resolver())
            self.cajas[clave] = tb
            if grupo:
                self.cromo.text(grupo[1], cap, grupo[0], color=UI["texto_tenue"], fontsize=6.8,
                                fontweight="bold", va="center")

        self.cromo.text(0.823, cap, "MÉTODO", color=UI["texto_tenue"], fontsize=6.8, fontweight="bold", va="center")
        self.boton_metodo = self._boton([0.823, y, 0.078, alto], self._texto_metodo(), self.siguiente_metodo,
                                        color_texto=GRAFICA["metodo"], fuente=FUENTE_MONO)
        self._etiquetar_boton_metodo()
        self.boton_resolver = self._boton([0.910, y, 0.082, alto], "▶  Resolver", self.resolver,
                                          color_texto=UI["ok"], acento=UI["ok"])

    def _controles(self) -> None:
        """Botones de reproducción en la cabecera de la gráfica."""
        alto = CABECERA - 0.010
        y = CUERPO_ARRIBA - CABECERA + 0.005
        x_fin = DER[0] + DER[1] - 0.006
        especificacion = [
            ("vista", "⟲ vista", 0.050, self.restablecer_vista),
            ("seguir", "◎ seguir", 0.055, self.alternar_seguir),
            ("fin", "▶|", 0.028, self.ir_al_final),
            ("siguiente", "▶", 0.026, self.paso_siguiente),
            ("animar", "▶ animar", 0.062, self.alternar_animacion),
            ("anterior", "◀", 0.026, self.paso_anterior),
            ("inicio", "|◀", 0.028, self.ir_al_inicio),
        ]
        self.botones: dict[str, Button] = {}
        x = x_fin
        for clave, texto, ancho, accion in especificacion:
            x -= ancho
            acento = UI["acento"] if clave == "animar" else None
            self.botones[clave] = self._boton([x, y, ancho, alto], texto, accion, acento=acento,
                                              color_texto=UI["acento"] if clave == "animar" else UI["texto"],
                                              tam=8.5)
            x -= 0.004

    def _boton(self, rect, texto, accion, color_texto=None, acento=None, fuente=FUENTE_UI, tam=9.5) -> Button:
        if acento:  # halo alrededor del botón
            for i, a in enumerate((0.10, 0.06, 0.035)):
                d = 0.0015 * (i + 1)
                self.cromo.add_patch(Rectangle((rect[0] - d, rect[1] - d * 1.7), rect[2] + 2 * d,
                                               rect[3] + 3.4 * d, color=acento, alpha=a, lw=0))
        ax = self.fig.add_axes(rect, zorder=3)
        # useblit=True: al pasar el ratón solo se repinta el botón (no toda la ventana).
        b = Button(ax, texto, color=UI["campo"], hovercolor=UI["campo_hover"], useblit=True)
        b.label.set_color(color_texto or UI["texto"])
        b.label.set_fontsize(tam)
        b.label.set_fontfamily(fuente)
        for s in ax.spines.values():
            s.set_color(acento or UI["borde"])
        b.on_clicked(lambda _evento: accion())
        return b

    def _barra_estado(self) -> None:
        ax = self.cromo
        ax.add_patch(Rectangle((0, 0), 1, ESTADO_ALTO, color=UI["estado"], lw=0))
        ax.add_patch(Rectangle((0, ESTADO_ALTO), 1, 0.0015, color=UI["borde"], lw=0))
        ax.add_patch(Rectangle((0, 0), 0.082, ESTADO_ALTO, color=UI["acento"], lw=0))
        yc = ESTADO_ALTO / 2
        ax.text(0.041, yc, "◆ EULER·LAB", color="#12001a", fontsize=8.5, fontweight="bold", va="center",
                ha="center")
        estilo = dict(fontsize=8, va="center", family=FUENTE_MONO)
        self.est_estado = ax.text(0.090, yc, "", color=UI["ok"], **estilo)
        self.est_metodo = ax.text(0.162, yc, "", color=UI["texto"], **estilo)
        self.est_param = ax.text(0.312, yc, "", color=UI["texto_medio"], **estilo)
        self.est_error = ax.text(0.462, yc, "", color=UI["texto_medio"], **estilo)
        self.est_coord = ax.text(0.612, yc, "", color=UI["texto_medio"], **estilo)
        ax.text(0.994, yc, "␣ animar  ←→ paso  rueda: zoom  arrastrar: mover  c: seguir",
                color=UI["texto_tenue"], ha="right", **{k: v for k, v in estilo.items()})

    # =================================================================== lógica principal
    def _leer_config(self) -> Configuracion:
        g = {k: tb.text.strip() for k, tb in self.cajas.items()}
        return Configuracion(ecuacion=g["ecuacion"], x0=g["x0"] or "0", y0=g["y0"] or "0", h=g["h"] or "0.1",
                             x_final=g["x_final"] or "1", metodo=self.metodo, exacta=g["exacta"],
                             max_pasos=self.config.max_pasos)

    def resolver(self, animar: bool = True) -> None:
        """Lee los campos, integra y (opcionalmente) empieza la animación desde el paso 0."""
        self.config = self._leer_config()
        self.animando = False
        try:
            self.sim = Simulacion(self.config)
        except (ErrorExpresion, KeyError, ValueError, ZeroDivisionError, OverflowError) as e:
            self.sim = None
            self._mostrar_error_entrada(e)
            self._redibujar()
            return
        self._error_txt = None
        self._poner_formula()
        self.grafica.cargar(self.sim)
        self.camara[:] = 0
        self._objetivo_vista = None
        self._preparar_registro()
        n = self.sim.n_pasos
        self.velocidad = float(np.clip(n / 8.0, 1.5, 6.0))
        if animar and n:
            self.revelados, self.seleccion, self.progreso = 1, 0, 0.0
            self.animando = True
            self._despertar()
        else:
            self.revelados, self.seleccion, self.progreso = n, max(n - 1, 0), 1.0
        self._actualizar_boton_animar()
        self._redibujar(completo=True)

    def _poner_formula(self) -> None:
        """Fórmula del método, coloreada, en la cabecera del panel de la gráfica."""
        for t in self.formula_textos:
            t.remove()
        met = self.sim.metodo
        tokens = [("ƒ  ", UI["texto_tenue"])] + [(t, CONSOLA.get(e, CONSOLA["texto"]))
                                                 for t, e in colorear(met.formula or met.nombre)]
        self.formula_textos = texto_coloreado(self.cromo, DER[0] + 0.104, CUERPO_ARRIBA - CABECERA / 2, tokens,
                                              fontsize=8.5, va="center")

    def _mostrar_error_entrada(self, e: Exception) -> None:
        texto = getattr(e, "texto", "") or ""
        pos = getattr(e, "posicion", None)
        msg = str(e).strip("'\"")
        lineas = [
            L(("# ✖ No se pudo resolver con estos datos", "error"), enfasis=True),
            L(),
        ]
        if texto:
            lineas.append(L(("entrada", "variable"), (" = ", "operador"), (f'"{texto}"', "cadena")))
            if pos is not None:
                lineas.append(L(" " * (len("entrada = \"") + pos), ("^", "error")))
        lineas += [L(("✖ ", "error"), (msg, "texto"))]
        for ayuda in ("", "# Ejemplos válidos:", "#   x + y      2x - 3y      y(1 - y/10)",
                      "#   sin(x)*y   e^(-x) - y   sqrt(abs(y))", "# Funciones: sin cos tan exp log sqrt abs ...",
                      "# Potencia con ^ o **; 2x se entiende como 2*x."):
            lineas.append(L((ayuda, "comentario")))
        self._lineas_error = lineas
        self._version_registro += 1
        self.consola.establecer(lineas)
        self.consola.seleccion = None
        self.consola.inicio = 0
        self.grafica.mostrar_error("No se pudo interpretar la entrada.\nRevisa el registro (izquierda).")
        self._error_txt = msg

    # ------------------------------------------------------------------ registro
    def _preparar_registro(self) -> None:
        """Construye una vez los renglones del encabezado y de cada paso."""
        s = self.sim
        met = s.metodo
        exacta = s.exacta.original if s.exacta else ""
        enc = [
            L(("# ── problema de valor inicial ", "comentario"), ("─" * 30, "comentario")),
            L(("edo", "variable"), ("      = ", "operador"), ('"', "cadena"), *colorear(f"y′ = {s.ecuacion.original}"),
              ('"', "cadena")),
            L(("x0, y0", "variable"), ("   = ", "operador"), (corto(s.x0), "numero"), (", ", "operador"),
              (corto(s.y0), "numero")),
            L(("h", "variable"), ("        = ", "operador"), (corto(s.h), "numero"),
              (f"      # {s.n_pasos} pasos hasta x = {corto(s.x_nodos[-1])}", "comentario")),
            L(("metodo", "variable"), ("   = ", "operador"), (str(met), "metodo"),
              (f"      # orden {met.orden}", "comentario")),
            L(("formula", "variable"), ("  : ", "operador"), *colorear(met.formula or "(sin fórmula)")),
            L(("ref", "variable"), ("      = ", "operador"),
              *(colorear(f"y(x) = {exacta}") if exacta else [(s.etiqueta_referencia, "comentario")])),
            L(("# " + "─" * 58, "comentario")),
        ]
        avisos_inicio = [a for a in s.avisos if a.nivel == "info"]
        for a in avisos_inicio:
            enc += self._renglones_aviso(a)
        enc.append(L())
        self._encabezado = enc

        bloques = []
        errores = s.errores
        inestables = set(s.pasos_inestables)
        for i, paso in enumerate(s.pasos):
            b = [L(("▸ ", "metodo"), ("paso", "clave"), (f" {i}", "numero"), ("   ", "texto"),
                   *colorear(f"(x{sub(i)}, y{sub(i)}) = ({num(paso.t)}, {num(paso.y)})"), paso=i, enfasis=True)]
            for etapa in paso.etapas:
                b.append(L("    ", *colorear(f"{etiqueta(etapa.etiqueta)} = f({num(etapa.t)}, {num(etapa.y)})"
                                             f" = {num(etapa.pendiente)}"), paso=i))
            formula = met.formula_numerica(paso)
            if formula:
                ancho = max(24, self.consola.columnas - 12)
                for renglon in formula.split("\n"):
                    for trozo in textwrap.wrap(renglon, ancho, subsequent_indent="      ", break_on_hyphens=False,
                                               drop_whitespace=False) or [""]:
                        b.append(L("    ", *colorear(trozo.rstrip()), paso=i))
            else:
                b.append(L("    ", *colorear(f"y{sub(i + 1)} = {num(paso.y_siguiente)}"), paso=i))
            e = errores[i + 1]
            fin = [("    ", "texto"), *colorear(f"x{sub(i + 1)} = {num(paso.t_siguiente)}")]
            if np.isfinite(e):
                fin += [("   |e", "tenue"), (sub(i + 1), "tenue"), ("| = ", "tenue"), (f"{e:.3e}", "error")]
            b.append(L(*fin, paso=i))
            if i in inestables:
                b.append(L("    ", ("⚠ inestable: el método amplifica el error en este paso", "aviso"), paso=i))
            b.append(L(paso=i))
            bloques.append(b)
        self._bloques = bloques

        fin = []
        if s.n_pasos:
            icono, palabra, estilo = ("✔ ", "listo", "ok") if s.completa else ("✖ ", "se detuvo", "error")
            fin.append(L((icono, estilo), (palabra, estilo), (f" · {s.n_pasos} pasos · ", "texto"),
                         *colorear(f"y({num(s.x_nodos[-1])}) ≈ {num(s.y_nodos[-1])}"), enfasis=True))
            e = errores[np.isfinite(errores)]
            if e.size:
                final = f"{errores[-1]:.3e}" if np.isfinite(errores[-1]) else "— (sin referencia ahí)"
                fin.append(L("  ", ("error final ", "tenue"), (final, "error"),
                             ("  ·  error máx ", "tenue"), (f"{e.max():.3e}", "error")))
            fin.append(L("  ", ("f evaluada ", "tenue"), (str(s.evaluaciones), "numero"), (" veces", "tenue")))
        for a in s.avisos:
            if a.nivel != "info":
                fin += self._renglones_aviso(a)
        self._final = fin
        self._version_registro += 1

    def _renglones_aviso(self, aviso) -> list[Linea]:
        # (ℹ no existe en DejaVu Sans Mono: se vería un cuadrito)
        icono, estilo = {"info": ("◉", "info"), "aviso": ("⚠", "aviso"), "error": ("✖", "error")}[aviso.nivel]
        ancho = max(20, self.consola.columnas - 10)
        palabras, renglones, actual = aviso.mensaje.split(), [], ""
        for p in palabras:
            if len(actual) + len(p) + 1 > ancho:
                renglones.append(actual)
                actual = p
            else:
                actual = f"{actual} {p}".strip()
        renglones.append(actual)
        salida = [L((f"{icono} ", estilo), (renglones[0], estilo))]
        salida += [L("  ", (r, estilo)) for r in renglones[1:]]
        return salida

    def _lineas_registro(self) -> list[Linea]:
        if self.sim is None:
            clave = ("error", self._version_registro)
        else:
            terminado = self.revelados >= len(self._bloques) and (self.progreso >= 1 and not self.animando)
            clave = (self._version_registro, self.revelados, terminado, self.animando)
        if clave == self._clave_lineas:
            return self._lineas_cache
        if self.sim is None:
            lineas = list(self._lineas_error)
        else:
            lineas = list(self._encabezado)
            for b in self._bloques[: self.revelados]:
                lineas += b
            if terminado:
                lineas += self._final
            elif self.animando:
                lineas.append(L(("❯ ", "metodo")))     # el cursor parpadeante es un artista aparte
        self._clave_lineas, self._lineas_cache = clave, lineas
        return lineas

    # ------------------------------------------------------------------ dibujo
    def refrescar(self) -> None:
        """Vuelve a pintar lo que haya cambiado (lo usan las pruebas y las capturas)."""
        self._redibujar()

    def _redibujar(self, completo: bool = False) -> None:
        """Actualiza los artistas según el estado y repinta lo que cambió.

        ``completo=True`` pide redibujar toda la ventana (solo cuando cambia la
        capa fija: textos de la barra superior, fórmula, tamaño...)."""
        self._actualizar_modelo()
        c = self.fig.canvas
        if (completo or self._est is None or self._tam != self._tam_actual()
                or not getattr(c, "supports_blit", False)):
            c.draw_idle()
            return
        self._componer(copiar_a_pantalla=True)

    def _actualizar_modelo(self) -> None:
        """Pasa el estado (paso, progreso, selección...) a los artistas, sin dibujar."""
        if self.sim is not None:
            self._pulso = (1.0 + 0.5 * math.sin((time.perf_counter() - self._t0) * 2 * math.pi * 1.2)
                           if self.animando and self.calidad.pulso else 1.0)
            self.grafica.mostrar(self.revelados, self.seleccion, min(self.progreso, 1.0), self._pulso)
            sel_antes = self.consola.seleccion
            self.consola.establecer(self._lineas_registro())
            self.consola.seleccion = self.seleccion if self.revelados else None
            if self.animando and (self.consola.pegado_al_final or sel_antes != self.seleccion):
                self.consola.ir_al_final()
        else:
            self.consola.establecer(self._lineas_registro())
        self._actualizar_estado()
        self._actualizar_cursor()

    def _actualizar_cursor(self) -> None:
        visible = False
        if self.animando and self.consola.lineas:
            pos = self.consola.coordenadas(len(self.consola.lineas) - 1, 2)
            if pos is not None:
                self.cursor.set_position(pos)
                visible = int((time.perf_counter() - self._t0) * 2.5) % 2 == 0
        self.cursor.set_visible(visible)

    # ---- capas
    def _tam_actual(self) -> tuple[int, int]:
        return round(self.fig.bbox.width), round(self.fig.bbox.height)

    def _regiones(self) -> dict:
        """Zonas de la ventana, en píxeles, que maneja cada capa."""
        principal = tuple(float(v) for v in self.grafica.capa.bbox.extents)
        franja = tuple(float(v) for v in self.grafica.capa_err.bbox.extents)
        clave = (self._tam_actual(), tuple(round(v, 2) for v in principal + franja))
        if self._regiones_cache is not None and self._regiones_cache[0] == clave:
            return self._regiones_cache[1]
        ancho, alto = self.fig.bbox.width, self.fig.bbox.height
        T = self.fig.transFigure

        def zona(x0, y0, x1, y1, pad=1):
            (a, b), (c, d) = T.transform([(x0, y0), (x1, y1)])
            return _caja_px(a, b, c, d, pad, ancho, alto)

        _, y_ax, _, _ = self.grafica.ax.get_position().bounds
        y_err, alto_err = self.grafica.ax_err.get_position().bounds[1::2]
        regiones = {
            "registro": zona(IZQ[0], ESTADO_ALTO + 0.006, IZQ[0] + IZQ[1], CUERPO_ARRIBA),
            "grafica": zona(DER[0], ESTADO_ALTO + 0.008, DER[0] + DER[1], CUERPO_ARRIBA - CABECERA),
            # parte de arriba (ejes principales con sus marcas) y de abajo (franja de error),
            # sin el renglón de la leyenda que queda entre ambas
            "sup": zona(DER[0], y_ax - 0.032, DER[0] + DER[1], CUERPO_ARRIBA - CABECERA, pad=0),
            "inf": zona(DER[0], ESTADO_ALTO + 0.008, DER[0] + DER[1], y_err + alto_err + 0.022, pad=0),
            "estado": zona(0, 0, 1, ESTADO_ALTO, pad=0),
            "principal": _caja_px(*principal, 2, ancho, alto),
            "franja": _caja_px(*franja, 2, ancho, alto),
        }
        self._regiones_cache = (clave, regiones)
        return regiones

    def _firmas_actuales(self) -> dict:
        """Lo que mostraría cada capa con el estado actual (si no cambió, no se repinta)."""
        g, c = self.grafica, self.consola
        vista = (*g.ax.get_xlim(), *g.ax.get_ylim())
        vista_err = (*g.ax_err.get_xlim(), *g.ax_err.get_ylim())
        return {
            "registro": (self._clave_lineas, c.inicio, c.seleccion, c.filas, self.txt_cabecera_registro.get_text()),
            "base": (g.version, vista, self.fondo.firma) + ((vista_err,) if "base_err" not in self._bases else ()),
            "base_err": (g.version, vista_err),
            "franja": (g.version, g.firma_base, vista_err),
            "dinamica": (g.version, vista, self.revelados, self.seleccion, round(self.progreso, 6),
                         round(self._pulso, 3)),
            "estado": tuple((t.get_text(), t.get_color()) for t in self._capa_estado),
            "cursor": (self.cursor.get_visible(), tuple(np.round(self.cursor.get_position(), 6))),
        }

    def _dibujar(self, artistas) -> None:
        for a in artistas:
            self.fig.draw_artist(a)

    def _pintar_registro(self, firma) -> None:
        c, R = self.fig.canvas, self._regiones()
        c.restore_region(self._est["registro"])
        self.consola.renderizar()
        self._dibujar(self._capa_registro)
        self._cache["registro"] = (firma, c.copy_from_bbox(R["registro"]))
        self._firmas["registro"] = firma
        self._firmas["cursor"] = None          # el cursor va encima: hay que volver a ponerlo

    _ENCIMA = {"principal": "dinamica", "franja": "franja"}

    def _pintar_base(self, nombre: str, firma) -> None:
        zona, artistas, _ = self._bases[nombre]
        self.fig.canvas.restore_region(self._est[zona])
        self._dibujar(artistas)
        self._guardar_base(nombre, firma)

    def _guardar_base(self, nombre: str, firma) -> None:
        """Guarda la zona con la base ya dibujada, y las sub-zonas donde se dibujan
        las capas que van encima (que entonces hay que volver a poner)."""
        c, R = self.fig.canvas, self._regiones()
        zona, _, encima = self._bases[nombre]
        self._cache[nombre] = (firma, c.copy_from_bbox(R[zona]))
        for sub in encima:
            self._cache[sub] = c.copy_from_bbox(R[sub])
            self._firmas[self._ENCIMA[sub]] = None
        self._firmas[nombre] = firma

    def _zona_de(self, sub: str) -> str:
        """Zona de la base sobre la que se dibuja la sub-zona ``sub``."""
        return next(zona for zona, _, encima in self._bases.values() if sub in encima)

    def _pintar_franja(self, firma) -> None:
        self.fig.canvas.restore_region(self._cache["franja"])
        self._dibujar(self._capa_franja)
        self._firmas["franja"] = firma

    def _pintar_dinamica(self, firma) -> None:
        self.fig.canvas.restore_region(self._cache["principal"])
        self._dibujar(self._capa_dinamica)
        self._firmas["dinamica"] = firma

    def _pintar_estado(self, firma) -> None:
        self.fig.canvas.restore_region(self._est["estado"])
        self._dibujar(self._capa_estado)
        self._firmas["estado"] = firma

    def _pintar_cursor(self, firma) -> None:
        self.fig.canvas.restore_region(self._cache["registro"][1])
        if self.cursor.get_visible():
            self.fig.draw_artist(self.cursor)
        self._firmas["cursor"] = firma

    def _componer(self, copiar_a_pantalla: bool = True) -> None:
        """Repinta solo las capas cuyo contenido cambió y copia a pantalla esas zonas."""
        R = self._regiones()
        f = self._firmas_actuales()
        zonas = []
        if f["registro"] != self._firmas.get("registro"):
            self._pintar_registro(f["registro"])
            zonas.append(R["registro"])
        for nombre, (zona, _, _) in self._bases.items():
            if f[nombre] != self._firmas.get(nombre):
                self._pintar_base(nombre, f[nombre])
                zonas.append(R[zona])
        if f["franja"] != self._firmas.get("franja"):
            self._pintar_franja(f["franja"])
            if R[self._zona_de("franja")] not in zonas:
                zonas.append(R["franja"])
        if f["dinamica"] != self._firmas.get("dinamica"):
            self._pintar_dinamica(f["dinamica"])
            if R[self._zona_de("principal")] not in zonas:
                zonas.append(R["principal"])
        if f["estado"] != self._firmas.get("estado"):
            self._pintar_estado(f["estado"])
            zonas.append(R["estado"])
        if f["cursor"] != self._firmas.get("cursor"):
            self._pintar_cursor(f["cursor"])
            if R["registro"] not in zonas:
                zonas.append(R["registro"])
        if copiar_a_pantalla:
            for z in zonas:
                self.fig.canvas.blit(z)

    def _al_dibujar(self, _evento) -> None:
        """Tras cada dibujo completo (que no incluye las capas): guardar las zonas de
        la capa fija y poner encima las demás, reutilizando lo ya dibujado."""
        if self._guardando:
            return
        c, R = self.fig.canvas, self._regiones()
        self._est = {k: c.copy_from_bbox(R[k]) for k in ("registro", "estado")}
        ancho, alto = self.fig.bbox.width, self.fig.bbox.height
        self._fotos_campos = {}
        for tb in self.cajas.values():      # cada campo tal como quedó (para repintarlo solo)
            zona = _caja_px(*tb.ax.bbox.extents, 2, ancho, alto)
            self._fotos_campos[id(tb)] = (zona, c.copy_from_bbox(zona))
        f = self._firmas_actuales()
        mismo_tam = self._tam == self._tam_actual()
        self._firmas = {}
        # Fondo de la gráfica: la imagen es lo más caro de la capa fija; se dibuja una vez.
        fondo = self._cache.get("fondo")
        if mismo_tam and fondo is not None:
            c.restore_region(fondo)
        else:
            self._dibujar(self._capa_fondo)
            fondo = c.copy_from_bbox(R["grafica"])
            self._cache["fondo"] = fondo
        self._est["grafica"] = fondo
        for sub in ("sup", "inf"):
            self._est[sub] = c.copy_from_bbox(R[sub])
        guardado = self._cache.get("registro")
        if mismo_tam and guardado and guardado[0] == f["registro"]:
            c.restore_region(guardado[1])
            self._firmas["registro"] = f["registro"]
        else:
            self._pintar_registro(f["registro"])
        for nombre, (_, _, encima) in self._bases.items():
            guardado = self._cache.get(nombre)
            if mismo_tam and guardado and guardado[0] == f[nombre] and all(s in self._cache for s in encima):
                c.restore_region(guardado[1])
                self._firmas[nombre] = f[nombre]
            else:
                self._pintar_base(nombre, f[nombre])
        self._tam = self._tam_actual()
        self._pintar_franja(f["franja"])
        self._pintar_dinamica(f["dinamica"])
        self._pintar_estado(f["estado"])
        self._pintar_cursor(f["cursor"])

    def _refrescar_estado(self) -> None:
        """Solo la barra de estado (p. ej. coordenadas del ratón): casi gratis."""
        if self._est is None or self._tam != self._tam_actual():
            return
        firma = tuple((t.get_text(), t.get_color()) for t in self._capa_estado)
        if firma != self._firmas.get("estado"):
            self._pintar_estado(firma)
            self.fig.canvas.blit(self._regiones()["estado"])

    def _invalidar(self) -> None:
        self._est = None
        self._cache.clear()
        self._firmas.clear()
        self._tam = None
        self._regiones_cache = None

    def _repintar_campo(self, campo, solo_probar: bool = False) -> bool:
        """Repinta un campo de texto sin redibujar la ventana. Parte de la foto del
        campo tomada en el último dibujo completo, así sus bordes no se acumulan."""
        foto = self._fotos_campos.get(id(campo)) if self._est is not None else None
        if foto is None or self._tam != self._tam_actual() or not getattr(self.fig.canvas, "supports_blit", False):
            return False
        if solo_probar:
            return True
        zona, region = foto
        self.fig.canvas.restore_region(region)
        visible = campo.label.get_visible()
        campo.label.set_visible(False)       # la etiqueta queda fuera del campo y ya está en la foto
        try:
            self.fig.draw_artist(campo.ax)
        finally:
            campo.label.set_visible(visible)
        self.fig.canvas.blit(zona)
        return True

    def _repintar_boton(self, boton) -> None:
        """Repinta un botón suelto (sin redibujar la ventana)."""
        if self._est is None:
            return                              # ya viene un dibujo completo
        self.fig.draw_artist(boton.ax)
        self.fig.canvas.blit(boton.ax.bbox)

    def _actualizar_estado(self) -> None:
        if self.sim is None:
            self.est_estado.set_text("✖ error")
            self.est_estado.set_color(UI["error"])
            self.est_metodo.set_text(str(obtener_metodo(self.metodo)))
            self.est_param.set_text("")
            self.est_error.set_text((self._error_txt or "")[:60])
            self.txt_cabecera_registro.set_text("")
            return
        s = self.sim
        hay_error = any(a.nivel == "error" for a in s.avisos)
        hay_aviso = any(a.nivel == "aviso" for a in s.avisos)
        if self.animando:
            self.est_estado.set_text("▶ animando")
            self.est_estado.set_color(UI["acento"])
        elif hay_error:
            self.est_estado.set_text("✖ se detuvo")
            self.est_estado.set_color(UI["error"])
        elif hay_aviso:
            self.est_estado.set_text("⚠ revisar")
            self.est_estado.set_color(UI["aviso"])
        else:
            self.est_estado.set_text("● listo")
            self.est_estado.set_color(UI["ok"])
        self.est_metodo.set_text(f"{s.metodo} · orden {s.metodo.orden}")
        self.est_param.set_text(f"h={corto(s.h)}  paso {self.seleccion}/{max(s.n_pasos - 1, 0)}  "
                                f"vel ×{self.velocidad / 2.5:.1f}")
        e = s.errores
        if self.revelados and np.isfinite(e[min(self.seleccion + 1, len(e) - 1)]):
            self.est_error.set_text(f"|e{sub(self.seleccion + 1)}| = {e[self.seleccion + 1]:.2e}   "
                                    f"evals f = {s.evaluaciones}")
        else:
            self.est_error.set_text(f"evals f = {s.evaluaciones}")
        self.txt_cabecera_registro.set_text(f"{self.revelados}/{s.n_pasos} pasos")

    # ------------------------------------------------------------------ animación
    def _despertar(self) -> None:
        """Arranca el temporizador si está detenido. (Si ya corre no se reinicia: con el
        ratón en movimiento llegarían eventos más rápido que los cuadros y nunca dispararía.)"""
        if self._timer is None:
            self._timer = self.fig.canvas.new_timer(interval=self.calidad.intervalo_ms)
            self._timer.add_callback(self._tick)
        if not self._timer_activo:
            self._timer_activo = True
            self._t_prev = time.perf_counter()
            self._timer.start()

    def _dormir(self) -> None:
        if self._timer is not None and self._timer_activo:
            self._timer.stop()
        self._timer_activo = False

    def _tick(self) -> None:
        ahora = time.perf_counter()
        dt = min(ahora - self._t_prev, 0.1)
        self._t_prev = ahora
        activo = False

        if self.animando and self.sim is not None:
            activo = True
            self.progreso += dt * self.velocidad
            if self.progreso >= 1 + PAUSA_EN_NODO:
                if self.revelados < self.sim.n_pasos:
                    self.revelados += 1
                    self.seleccion = self.revelados - 1
                    self.progreso = 0.0
                else:
                    self.progreso = 1.0
                    self.animando = False
                    self.consola.ir_al_final()
                    self._actualizar_boton_animar()

        activo |= self._mover_camara(dt)

        if self.calidad.parallax:       # parallax del ratón, con suavizado
            delta = self.raton - self.raton_suave
            if np.abs(delta).max() > 1e-3:
                self.raton_suave += delta * (1 - math.exp(-dt * 7))
                activo = True
            self.fondo.actualizar(self.camara, self.raton_suave, t=ahora - self._t0 if self.animando else None)
        self._redibujar()
        if not activo and not self._sucio:
            self._dormir()
        self._sucio = False

    def _mover_camara(self, dt: float) -> bool:
        """Acerca la vista a su objetivo (modo seguir o regreso a la vista inicial)."""
        if self.sim is None:
            return False
        ax = self.grafica.ax
        objetivo = None
        if self.seguir and self.sim.pasos:
            (hx0, hx1), (hy0, hy1) = self.grafica.vista_inicial
            cx, cy = self.grafica.punto_actual(self.revelados, self.seleccion, min(self.progreso, 1.0))
            ax_, ay_ = (hx1 - hx0) * 0.42, (hy1 - hy0) * 0.42
            objetivo = np.array([cx - ax_ / 2, cx + ax_ / 2, cy - ay_ / 2, cy + ay_ / 2])
        elif self._objetivo_vista is not None:
            objetivo = np.array(self._objetivo_vista)
        if objetivo is None:
            return False
        actual = np.array([*ax.get_xlim(), *ax.get_ylim()])
        nuevo = actual + (objetivo - actual) * (1 - math.exp(-dt * 4.5))
        listo = np.abs(nuevo - objetivo).max() < 1e-4 * max(np.abs(objetivo).max(), 1)
        if listo:
            nuevo = objetivo
            if not self.seguir:
                self._objetivo_vista = None
        self._fijar_vista(nuevo[:2], nuevo[2:])
        return not listo

    def _fijar_vista(self, xlim, ylim) -> None:
        self.grafica.aplicar_vista(tuple(xlim), tuple(ylim))
        (hx0, hx1), (hy0, hy1) = self.grafica.vista_inicial
        cx, cy = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        self.camara[:] = ((cx - (hx0 + hx1) / 2) / (hx1 - hx0), (cy - (hy0 + hy1) / 2) / (hy1 - hy0))
        self.fondo.actualizar(self.camara, self.raton_suave)

    # =================================================================== acciones (botones y teclas)
    def _actualizar_boton_animar(self) -> None:
        texto = "❚❚ pausa" if self.animando else "▶ animar"
        boton = self.botones["animar"]
        if boton.label.get_text() != texto:
            boton.label.set_text(texto)
            self._repintar_boton(boton)

    def alternar_animacion(self) -> None:
        if self.sim is None or not self.sim.n_pasos:
            return
        if self.animando:
            self.animando = False
        else:
            if self.revelados >= self.sim.n_pasos and self.progreso >= 1:
                self.revelados, self.seleccion, self.progreso = 1, 0, 0.0   # volver a empezar
            else:
                self.seleccion = self.revelados - 1
                if self.progreso >= 1:
                    self.progreso = 1.0
            self.animando = True
            self._despertar()
        self._actualizar_boton_animar()
        self._redibujar()

    def paso_siguiente(self) -> None:
        if self.sim is None or not self.sim.n_pasos:
            return
        self.animando = False
        if self.seleccion < self.revelados - 1:
            self.seleccion += 1
        elif self.progreso < 1:
            self.progreso = 1.0
        elif self.revelados < self.sim.n_pasos:
            self.revelados += 1
            self.seleccion = self.revelados - 1
            self.progreso = 1.0
        self.consola.asegurar_visible(self.seleccion)
        self._tras_navegar()

    def paso_anterior(self) -> None:
        if self.sim is None or not self.sim.n_pasos:
            return
        self.animando = False
        self.progreso = 1.0
        self.seleccion = max(0, self.seleccion - 1)
        self._tras_navegar(asegurar=True)

    def ir_al_inicio(self) -> None:
        if self.sim is None or not self.sim.n_pasos:
            return
        self.animando = False
        self.revelados, self.seleccion, self.progreso = 1, 0, 1.0
        self.consola.inicio = 0
        self._tras_navegar()

    def ir_al_final(self) -> None:
        if self.sim is None:
            return
        self.animando = False
        n = self.sim.n_pasos
        self.revelados, self.seleccion, self.progreso = n, max(n - 1, 0), 1.0
        self.consola.ir_al_final()
        self._tras_navegar()

    def _tras_navegar(self, asegurar: bool = False) -> None:
        self._actualizar_boton_animar()
        self.consola.establecer(self._lineas_registro())
        if asegurar:
            self.consola.asegurar_visible(self.seleccion)
        if self.seguir:
            self._despertar()
        self._redibujar()

    def alternar_seguir(self) -> None:
        self.seguir = not self.seguir
        b = self.botones["seguir"]
        b.color = "#1d1440" if self.seguir else UI["campo"]
        b.ax.set_facecolor(b.color)
        b.label.set_color(UI["acento"] if self.seguir else UI["texto"])
        self._repintar_boton(b)
        if not self.seguir:
            self._objetivo_vista = self.grafica.vista_inicial[0] + self.grafica.vista_inicial[1]
        self._despertar()
        self._redibujar()

    def restablecer_vista(self) -> None:
        if self.seguir:
            self.alternar_seguir()
        self._objetivo_vista = self.grafica.vista_inicial[0] + self.grafica.vista_inicial[1]
        self._despertar()

    def siguiente_metodo(self) -> None:
        i = self.metodos.index(self.metodo) if self.metodo in self.metodos else -1
        self.metodo = self.metodos[(i + 1) % len(self.metodos)]
        self._etiquetar_boton_metodo()
        self.resolver()

    def _etiquetar_boton_metodo(self) -> None:
        texto = self._texto_metodo()
        self.boton_metodo.label.set_text(texto)
        self.boton_metodo.label.set_fontsize(9.5 if len(texto) <= 11 else max(6.5, 9.5 * 11 / len(texto)))

    def _texto_metodo(self) -> str:
        return f"{self.metodo} ▾"

    # =================================================================== eventos
    def _tecleando(self) -> bool:
        return any(tb.capturekeystrokes for tb in self.cajas.values())

    def _al_teclear(self, evento) -> None:
        if self._tecleando():
            return
        acciones = {
            " ": self.alternar_animacion, "right": self.paso_siguiente, "left": self.paso_anterior,
            "home": self.ir_al_inicio, "end": self.ir_al_final, "c": self.alternar_seguir,
            "r": self.restablecer_vista, "m": self.siguiente_metodo, "enter": self.resolver,
        }
        if evento.key in acciones:
            acciones[evento.key]()
        elif evento.key in ("+", "="):
            self.velocidad = min(self.velocidad * 1.4, 30)
            self._redibujar()
        elif evento.key == "-":
            self.velocidad = max(self.velocidad / 1.4, 0.3)
            self._redibujar()
        elif evento.key == "up":
            self.consola.desplazar(-3)
            self._redibujar()
        elif evento.key == "down":
            self.consola.desplazar(3)
            self._redibujar()

    def _al_rodar(self, evento) -> None:
        if evento.inaxes is self.consola.ax:
            self.consola.desplazar(-3 * int(np.sign(evento.step) or 1) * max(1, abs(int(evento.step))))
            self._redibujar()
            return
        en_grafica = self.grafica.contiene(evento.inaxes)
        if (en_grafica or self.grafica.en_franja(evento.inaxes)) and self.sim is not None:
            if self.seguir:
                self.alternar_seguir()
            self._objetivo_vista = None
            factor = 0.85 ** evento.step
            ax = self.grafica.ax
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            cx = evento.xdata
            xlim = (cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
            if en_grafica:
                cy = evento.ydata
                ylim = (cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
            else:
                ylim = (y0, y1)
            self._fijar_vista(xlim, ylim)
            self._pedir_cuadro()   # se pinta en el siguiente cuadro (agrupa los giros rápidos de la rueda)

    def _pedir_cuadro(self) -> None:
        """Marca que hay algo que pintar; el temporizador lo hace a su ritmo."""
        self._sucio = True
        self._despertar()

    def _al_presionar(self, evento) -> None:
        for tb in self.cajas.values():   # borde cian en el campo que tiene el foco
            color = UI["borde_activo"] if tb.capturekeystrokes else UI["borde"]
            cambio = False
            for borde in tb.ax.spines.values():
                if tuple(borde.get_edgecolor()) != to_rgba(color):
                    borde.set_color(color)
                    cambio = True
            if cambio and not self._repintar_campo(tb):
                self.fig.canvas.draw_idle()
        if evento.inaxes is self.consola.ax and evento.button == 1:
            paso = self.consola.paso_en(evento.ydata)
            if paso is not None:
                self.animando = False
                self.seleccion = paso
                if paso == self.revelados - 1:
                    self.progreso = 1.0
                self._tras_navegar()
            return
        if self.grafica.contiene(evento.inaxes) and self.sim is not None:
            if evento.dblclick:
                self.restablecer_vista()
                return
            if evento.button == 1:
                if self.seguir:
                    self.alternar_seguir()
                self._objetivo_vista = None
                ax = self.grafica.ax
                self._arrastre = (evento.x, evento.y, ax.get_xlim(), ax.get_ylim())

    def _al_soltar(self, _evento) -> None:
        self._arrastre = None

    def _al_mover(self, evento) -> None:
        if self.calidad.parallax:
            # Parallax: posición del ratón respecto al centro del panel de la gráfica.
            x, w = DER
            y0 = ESTADO_ALTO
            fx = evento.x / self.fig.bbox.width
            fy = evento.y / self.fig.bbox.height
            if x <= fx <= x + w and y0 <= fy <= CUERPO_ARRIBA:
                self.raton[:] = (np.clip((fx - x) / w - 0.5, -0.5, 0.5),
                                 np.clip((fy - y0) / (CUERPO_ARRIBA - y0) - 0.5, -0.5, 0.5))
                self._despertar()

        if self._arrastre is not None and self.sim is not None:
            px, py, (x0, x1), (y0_, y1_) = self._arrastre
            caja = self.grafica.ax.get_window_extent()
            dx = (evento.x - px) * (x1 - x0) / caja.width
            dy = (evento.y - py) * (y1_ - y0_) / caja.height
            self._fijar_vista((x0 - dx, x1 - dx), (y0_ - dy, y1_ - dy))
            self._pedir_cuadro()

        # Coordenadas en la barra de estado: solo se repinta esa barra.
        if self.grafica.contiene(evento.inaxes) and evento.xdata is not None:
            texto = f"x = {evento.xdata: .4f}   y = {evento.ydata: .4f}"
        else:
            texto = ""
        if texto != self.est_coord.get_text():
            self.est_coord.set_text(texto)
            self._refrescar_estado()

    def _al_salir(self, _evento) -> None:
        if self.est_coord.get_text():
            self.est_coord.set_text("")
            self._refrescar_estado()
        if self.calidad.parallax:
            self.raton[:] = 0
            self._despertar()

    def _al_redimensionar(self, _evento) -> None:
        self._invalidar()
        self.consola.medir()
        if self.sim is not None:
            self._preparar_registro()
            self.grafica.actualizar_campo()
        self._redibujar(completo=True)

    # =================================================================== utilidades públicas
    def ir_a(self, revelados: int, seleccion: Optional[int] = None, progreso: float = 1.0) -> "EulerLab":
        """Fija el estado sin animar (útil para capturas y pruebas)."""
        self.animando = False
        if self.sim is not None:
            n = self.sim.n_pasos
            self.revelados = max(0, min(revelados, n))
            self.seleccion = max(0, min(self.revelados - 1 if seleccion is None else seleccion, self.revelados - 1))
            self.progreso = progreso
        self._actualizar_boton_animar()
        self.consola.establecer(self._lineas_registro())
        self.consola.asegurar_visible(self.seleccion)
        self._redibujar()
        return self

    def guardar(self, ruta: str, dpi: int = 110) -> str:
        """Guarda una captura de la ventana (incluidas las capas)."""
        self._actualizar_modelo()
        self._guardando = True
        for a in self._animados:
            a.set_animated(False)
        try:
            self.consola.renderizar()
            self.fig.savefig(ruta, dpi=dpi, facecolor=self.fig.get_facecolor())
        finally:
            for a in self._animados:
                a.set_animated(True)
            self._guardando = False
            self._invalidar()      # el renderer pudo cambiar de tamaño: el próximo dibujo es completo
        return ruta

    def mostrar(self) -> None:
        import matplotlib.pyplot as plt

        backend = matplotlib.get_backend().lower()
        if "inline" in backend or backend in ("agg", "pdf", "ps", "svg", "cairo", "template"):
            print(f"Aviso: el backend de matplotlib '{backend}' no abre ventanas, así que la app no será "
                  "interactiva.\n  · Ejecútala desde una terminal:  python euler_app.py\n"
                  "  · En Jupyter: %matplotlib qt  (o %matplotlib widget con ipympl) antes de crearla.\n"
                  "  · En Linux quizá falte Tk:  sudo apt install python3-tk")
        plt.show()


def corto_o_texto(valor) -> str:
    """Deja las expresiones como texto (``"pi/2"``) y formatea los números."""
    if isinstance(valor, str):
        return valor
    return corto(valor)


def texto_coloreado(ax, x, y, tokens, **kw):
    """Texto de varios colores en una línea con fuente monoespaciada.

    ``tokens`` = [(texto, color), ...]. Se crea un ``Text`` por color, todos en
    la misma posición, reemplazando con espacios lo que es de otro color.
    """
    textos = []
    for color in dict.fromkeys(c for _, c in tokens):
        cadena = "".join(t if c == color else " " * len(t) for t, c in tokens).rstrip()
        textos.append(ax.text(x, y, cadena, color=color, family=FUENTE_MONO, parse_math=False, **kw))
    return textos


def _brillo(color, ancho=4.0, alpha=0.35):
    from matplotlib import patheffects

    return [patheffects.withStroke(linewidth=ancho, foreground=color, alpha=alpha), patheffects.Normal()]


def main(config: Optional[Configuracion] = None, calidad: str = "media") -> EulerLab:
    """Abre la app. Usada por ``python -m edo.app`` y por ``euler_app.py``."""
    app = EulerLab(config, calidad=calidad)
    app.mostrar()
    return app
