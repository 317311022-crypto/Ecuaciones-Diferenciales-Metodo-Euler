"""Ventana principal: barra de entradas arriba, registro a la izquierda y
gráfica a la derecha.

Todo está hecho con matplotlib (sin dependencias extra): los campos de texto
y botones son ``matplotlib.widgets``, el registro es :class:`Consola` y la
gráfica :class:`PanelGrafica`. Un temporizador de ~30 fps anima el paso a
paso, la cámara y el parallax, y se apaga solo cuando no hay nada moviéndose.

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
import textwrap
import time
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox

from ..expresion import ErrorExpresion
from ..formato import corto, etiqueta, num, sub
from ..metodos import metodos_disponibles, obtener_metodo
from ..simulacion import Configuracion, Simulacion
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


def _orden_metodos() -> list[str]:
    nombres = metodos_disponibles()
    return sorted(nombres, key=lambda n: (n != "euler", n))


class EulerLab:
    """La aplicación. ``EulerLab(Configuracion(...)).mostrar()``."""

    def __init__(self, config: Optional[Configuracion] = None, figsize=(15.5, 8.7), dpi: int = 100,
                 animar: bool = True):
        aplicar_tema(app=True)
        import matplotlib.pyplot as plt

        self.config = config or Configuracion()
        self.fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=UI["fondo"])
        try:
            self.fig.canvas.manager.set_window_title("Euler·Lab — método de Euler paso a paso")
        except AttributeError:
            pass

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

        self._cromo()
        self.fondo = FondoParallax(self.fig, self._rect_grafica_panel(), zorder=-5)
        x, w = DER
        self.grafica = PanelGrafica(self.fig, [x + 0.040, 0.318, w - 0.052, 0.500],
                                    [x + 0.040, 0.075, w - 0.052, 0.168])
        x, w = IZQ
        self.consola = Consola(self.fig, [x + 0.002, ESTADO_ALTO + 0.010, w - 0.004,
                                          CUERPO_ARRIBA - CABECERA - ESTADO_ALTO - 0.012])
        self._entradas()
        self._controles()
        self._barra_estado()

        # Lo que cambia en cada cuadro se dibuja aparte ("blitting"): el resto de la
        # ventana se guarda como imagen y solo se repinta cuando de verdad cambia.
        self.cursor = self.consola.ax.text(0, 0, "▌", color=CONSOLA["titulo"], family=FUENTE_MONO,
                                           fontsize=self.consola.tam, va="baseline", visible=False, zorder=3)
        self._animados = [self.fondo.ax, self.grafica.ax, self.grafica.ax_err, self.cursor,
                          self.est_estado, self.est_metodo, self.est_param, self.est_error, self.est_coord,
                          self.txt_cabecera_registro]
        for a in self._animados:
            a.set_animated(True)
        self._fondo_blit = None
        self._firma = None
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
        texto_coloreado(self.fig, DER[0] + 0.040, 0.2815, tokens, fontsize=7.8, va="center", zorder=5)
        self.formula_textos = []

    def _entradas(self) -> None:
        """Campos de la parte superior: ecuación, condición inicial, paso, intervalo..."""
        c = self.config
        y, alto = ENTRADAS_Y + 0.010, 0.040
        cap = ENTRADAS_Y + 0.060
        self.cajas: dict[str, TextBox] = {}
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
            tb = TextBox(ax, etiq + " ", initial=str(valor), color=UI["campo"], hovercolor=UI["campo_hover"],
                         label_pad=6 / ancho_px)
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
        # useblit=False: el resaltado al pasar el ratón provoca un redibujo completo, así
        # el fondo que guardamos para el blitting nunca queda desactualizado.
        b = Button(ax, texto, color=UI["campo"], hovercolor=UI["campo_hover"], useblit=False)
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

    def _renglones_aviso(self, aviso) -> list[Linea]:
        icono, estilo = {"info": ("ℹ", "info"), "aviso": ("⚠", "aviso"), "error": ("✖", "error")}[aviso.nivel]
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
            return list(self._lineas_error)
        lineas = list(self._encabezado)
        for b in self._bloques[: self.revelados]:
            lineas += b
        terminado = self.revelados >= len(self._bloques) and (self.progreso >= 1 and not self.animando)
        if terminado:
            lineas += self._final
        elif self.animando:
            lineas.append(L(("❯ ", "metodo")))     # el cursor parpadeante es un artista aparte
        return lineas

    # ------------------------------------------------------------------ dibujo
    def _redibujar(self, completo: bool = False) -> None:
        """Actualiza los artistas según el estado y repinta: completo si cambió algo
        de la parte "fija" (registro, botones), rápido (blit) si no."""
        if self.sim is not None:
            pulso = 1.0 + (0.5 * math.sin((time.perf_counter() - self._t0) * 2 * math.pi * 1.2)
                           if self.animando else 0.0)
            self.grafica.mostrar(self.revelados, self.seleccion, min(self.progreso, 1.0), pulso)
            sel_antes = self.consola.seleccion
            self.consola.establecer(self._lineas_registro())
            self.consola.seleccion = self.seleccion if self.revelados else None
            if self.animando and (self.consola.pegado_al_final or sel_antes != self.seleccion):
                self.consola.ir_al_final()
        self._actualizar_estado()
        firma = (len(self.consola.lineas), self.consola.inicio, self.consola.seleccion, self.consola.filas,
                 id(self.sim), self._error_txt, self.botones["animar"].label.get_text())
        self._actualizar_cursor()
        if completo or firma != self._firma or self._fondo_blit is None:
            self._firma = firma
            self.consola.renderizar()
            self.fig.canvas.draw_idle()
        else:
            self._refrescar_rapido()

    def _actualizar_cursor(self) -> None:
        visible = False
        if self.animando and self.consola.lineas:
            pos = self.consola.coordenadas(len(self.consola.lineas) - 1, 2)
            if pos is not None:
                self.cursor.set_position(pos)
                visible = int((time.perf_counter() - self._t0) * 2.5) % 2 == 0
        self.cursor.set_visible(visible)

    def _dibujar_animados(self) -> None:
        for a in self._animados:
            self.fig.draw_artist(a)

    def _al_dibujar(self, _evento) -> None:
        """Tras cada dibujo completo: guardar el fondo y pintar encima lo animado."""
        if self._guardando:
            return
        self._fondo_blit = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._dibujar_animados()

    def _refrescar_rapido(self) -> None:
        c = self.fig.canvas
        if self._fondo_blit is None:
            c.draw_idle()
            return
        c.restore_region(self._fondo_blit)
        self._dibujar_animados()
        c.blit(self.fig.bbox)

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
            self._timer = self.fig.canvas.new_timer(interval=33)
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

        # Parallax del ratón con suavizado.
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
        self.botones["animar"].label.set_text("❚❚ pausa" if self.animando else "▶ animar")

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
        if not self.seguir:
            self._objetivo_vista = self.grafica.vista_inicial[0] + self.grafica.vista_inicial[1]
        self._despertar()
        self._redibujar(completo=True)

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
        if evento.inaxes in (self.grafica.ax, self.grafica.ax_err) and self.sim is not None:
            if self.seguir:
                self.alternar_seguir()
            self._objetivo_vista = None
            factor = 0.85 ** evento.step
            ax = self.grafica.ax
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            cx = evento.xdata
            xlim = (cx + (x0 - cx) * factor, cx + (x1 - cx) * factor)
            if evento.inaxes is ax:
                cy = evento.ydata
                ylim = (cy + (y0 - cy) * factor, cy + (y1 - cy) * factor)
            else:
                ylim = (y0, y1)
            self._fijar_vista(xlim, ylim)
            self._redibujar()

    def _al_presionar(self, evento) -> None:
        for tb in self.cajas.values():   # borde cian en el campo que tiene el foco
            for s in tb.ax.spines.values():
                s.set_color(UI["borde_activo"] if tb.capturekeystrokes else UI["borde"])
        if evento.inaxes is self.consola.ax and evento.button == 1:
            paso = self.consola.paso_en(evento.ydata)
            if paso is not None:
                self.animando = False
                self.seleccion = paso
                if paso == self.revelados - 1:
                    self.progreso = 1.0
                self._tras_navegar()
            return
        if evento.inaxes is self.grafica.ax and self.sim is not None:
            if evento.dblclick:
                self.restablecer_vista()
                return
            if evento.button == 1:
                if self.seguir:
                    self.alternar_seguir()
                self._objetivo_vista = None
                ax = self.grafica.ax
                self._arrastre = (evento.x, evento.y, ax.get_xlim(), ax.get_ylim())
        self.fig.canvas.draw_idle()

    def _al_soltar(self, _evento) -> None:
        self._arrastre = None

    def _al_mover(self, evento) -> None:
        # Parallax: posición del ratón respecto al centro del panel de la gráfica.
        x, w = DER
        y0 = ESTADO_ALTO
        fx = evento.x / (self.fig.get_figwidth() * self.fig.dpi)
        fy = evento.y / (self.fig.get_figheight() * self.fig.dpi)
        if x <= fx <= x + w and y0 <= fy <= CUERPO_ARRIBA:
            self.raton[:] = (np.clip((fx - x) / w - 0.5, -0.5, 0.5), np.clip((fy - y0) / (CUERPO_ARRIBA - y0) - 0.5,
                                                                            -0.5, 0.5))
            self._despertar()

        if self._arrastre is not None and self.sim is not None:
            px, py, (x0, x1), (y0_, y1_) = self._arrastre
            caja = self.grafica.ax.get_window_extent()
            dx = (evento.x - px) * (x1 - x0) / caja.width
            dy = (evento.y - py) * (y1_ - y0_) / caja.height
            self._fijar_vista((x0 - dx, x1 - dx), (y0_ - dy, y1_ - dy))
            self._redibujar()

        if evento.inaxes is self.grafica.ax and evento.xdata is not None:
            self.est_coord.set_text(f"x = {evento.xdata: .4f}   y = {evento.ydata: .4f}")
            self._sucio = True
            self._despertar()
        elif self.est_coord.get_text():
            self.est_coord.set_text("")
            self._sucio = True
            self._despertar()

    def _al_salir(self, _evento) -> None:
        self.raton[:] = 0
        self._despertar()

    def _al_redimensionar(self, _evento) -> None:
        self._fondo_blit = None
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
        """Guarda una captura de la ventana (incluidas las partes animadas)."""
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
            self._fondo_blit = None      # el renderer pudo cambiar de tamaño: forzar un dibujo completo
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


def main(config: Optional[Configuracion] = None) -> EulerLab:
    """Abre la app. Usada por ``python -m edo.app`` y por ``euler_app.py``."""
    app = EulerLab(config)
    app.mostrar()
    return app
