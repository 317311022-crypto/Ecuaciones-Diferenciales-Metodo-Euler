"""Panel derecho: la gráfica del paso a paso.

Está dividida en dos ejes superpuestos que comparten límites, para que en cada
cuadro de la animación se redibuje lo mínimo:

**Base** (``ax`` y la franja ``ax_err``), cambia solo al mover la vista o al
cambiar de paso:

1. Campo de pendientes de ``y' = f(x, y)``: un segmentito con la pendiente de
   cada punto. Se recalcula al mover la vista, así siempre cubre lo visible.
2. Solución de referencia (exacta o RK4 fino), en cian.
3. Ejes, rejilla y la franja con el error global ``|y(xₙ) − yₙ|`` de cada nodo.

**Capa** (``capa``, transparente, encima), cambia en cada cuadro:

4. Poligonal del método con su halo, el resplandor inferior y los nodos.
5. Geometría del paso seleccionado: recta tangente, triángulo ``h`` / ``h·m``,
   las etapas ``k₁, k₂...`` (puntos donde el método evaluó ``f``), el punto que
   avanza y el error contra la referencia.

Todo lo de la capa se recorta al rectángulo de los ejes, así la ventana puede
copiar a la pantalla solo esa zona.
"""

from __future__ import annotations

import numpy as np
from matplotlib import patheffects
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator

from ..formato import etiqueta as etiqueta_sub
from ..formato import num, sub
from .calidad import CALIDADES, Calidad
from .efectos import DegradadoBajoCurva, PuntosLuminosos, TrazoLuminoso, mezclar
from .tema import FUENTE_MONO, GRAFICA, UI, color_etapa

MAX_ETAPAS = 6
MAX_NODOS_CON_MARCA = 150      # con más nodos visibles, las marcas solo estorban (y cuestan)

_CAMPO = LinearSegmentedColormap.from_list(
    "campo", [GRAFICA["campo_bajo"], GRAFICA["campo_medio"], GRAFICA["campo_alto"]])


def _brillo_texto(color, ancho=3.0, alpha=0.35):
    return [patheffects.withStroke(linewidth=ancho, foreground=color, alpha=alpha), patheffects.Normal()]


def _caja(color, alpha_borde=0.55):
    """Cajita oscura tipo 'tooltip' detrás de una etiqueta (legible sobre cualquier fondo)."""
    return dict(boxstyle="round,pad=0.28,rounding_size=0.25", facecolor=(0.02, 0.03, 0.07, 0.82),
                edgecolor=to_rgba(color, alpha_borde), linewidth=0.7)


def _estilizar_ejes(ax, abajo=True):
    ax.patch.set_alpha(0)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRAFICA["eje"])
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=GRAFICA["ticks"], labelsize=7.5, length=3, width=0.6, labelbottom=abajo)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily(FUENTE_MONO)
    ax.grid(True, color=GRAFICA["rejilla"], lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_autoscale_on(False)


class PanelGrafica:
    def __init__(self, fig, rect_principal, rect_error, calidad: Calidad = CALIDADES["media"]):
        self.fig = fig
        self.q = calidad
        self.sim = None
        self.version = 0                 # cambia con cada simulación cargada
        self.firma_base: tuple = ()      # estado de la franja de error (para saber si hay que redibujarla)

        # ---------------------------------------------------------------- base
        self.ax = fig.add_axes(rect_principal, zorder=2)
        # La franja de error es una vista general de TODO el intervalo: no sigue el zoom
        # de la gráfica (así la cámara "seguir" no obliga a redibujarla en cada cuadro).
        self.ax_err = fig.add_axes(rect_error, zorder=2)
        _estilizar_ejes(self.ax)
        _estilizar_ejes(self.ax_err)
        self.ax.yaxis.set_major_locator(MaxNLocator(7))
        self.ax_err.yaxis.set_major_locator(MaxNLocator(3))
        self.ax_err.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        self.ax_err.yaxis.get_offset_text().set_color(GRAFICA["ticks"])
        self.ax_err.yaxis.get_offset_text().set_fontsize(7)
        ax = self.ax
        H = self.q.halo

        ax.text(0.004, 0.985, "y", transform=ax.transAxes, color=UI["texto_medio"], fontsize=9,
                family=FUENTE_MONO, va="top")
        self.ax_err.text(0.997, 0.97, "error global  |y(xₙ) − yₙ|", transform=self.ax_err.transAxes,
                         color=UI["texto_tenue"], fontsize=7.5, family=FUENTE_MONO, va="top", ha="right")
        self.ax_err.text(0.997, 0.04, "x", transform=self.ax_err.transAxes, color=UI["texto_medio"],
                         fontsize=9, family=FUENTE_MONO, ha="right")

        self.campo = LineCollection([], linewidths=1.15, capstyle="round", zorder=1)
        ax.add_collection(self.campo)
        n, a = H(6, 0.028)
        self.ref = TrazoLuminoso(ax, GRAFICA["referencia"], lw=1.5, capas=n, alcance=9, intensidad=a,
                                 nucleo=0.35, zorder=4)
        self.ref_txt = ax.annotate("", (0, 0), xytext=(-6, 8), textcoords="offset points", color=GRAFICA["referencia"],
                                   fontsize=7.5, family=FUENTE_MONO, ha="right", va="bottom", zorder=8,
                                   annotation_clip=True, clip_on=True, bbox=_caja(GRAFICA["referencia"], 0.35))
        self.mensaje = ax.text(0.5, 0.5, "", transform=ax.transAxes, ha="center", va="center", fontsize=11,
                               color=UI["error"], family=FUENTE_MONO, zorder=20, wrap=True,
                               path_effects=_brillo_texto(UI["error"], 5, 0.2))

        # Franja de error: las barras van en una capa propia encima de sus ejes (``capa_err``),
        # así revelar o seleccionar un paso no obliga a redibujar ejes y marcas.
        self.capa_err = fig.add_axes(rect_error, zorder=2.5, sharex=self.ax_err, sharey=self.ax_err)
        self.capa_err.set_axis_off()
        self.capa_err.set_autoscale_on(False)
        self.capa_err.set_navigate(False)
        self.barras = PolyCollection([], facecolors=to_rgba(GRAFICA["error"], 0.28),
                                     edgecolors=to_rgba(GRAFICA["error"], 0.7), linewidths=0.6, zorder=3)
        self.capa_err.add_collection(self.barras)
        self._verts_barras = np.empty((0, 4, 2))
        n, al = self.q.halo_puntos(4, 2.6)
        self.err_punto = PuntosLuminosos(self.capa_err, GRAFICA["error"], tam=22, capas=n, alcance=al,
                                         intensidad=0.12, zorder=6)
        self.err_avisos = self.capa_err.scatter([], [], marker="^", s=26, color=UI["aviso"], zorder=7,
                                                linewidths=0)

        # ---------------------------------------------------------------- capa (se anima)
        self.capa = fig.add_axes(rect_principal, zorder=2.5, sharex=self.ax, sharey=self.ax)
        self.capa.set_axis_off()
        self.capa.set_autoscale_on(False)
        self.capa.set_navigate(False)
        c = self.capa

        self.degradado = None
        if self.q.resplandor:
            franjas = (0.025, 0.06, 0.11, 0.19) if self.q.brillo >= 1 else (0.04, 0.12)
            self.degradado = DegradadoBajoCurva(c, GRAFICA["metodo"], alpha=0.045 * 4 / len(franjas) * 0.8,
                                                franjas=franjas, zorder=2.5)
        n, a = H(9, 0.045)
        self.poligonal = TrazoLuminoso(c, GRAFICA["metodo"], lw=2.1, capas=n, alcance=15, intensidad=a,
                                       nucleo=0.6, zorder=6)
        n, al = self.q.halo_puntos(3, 2.4)
        self.nodos = PuntosLuminosos(c, GRAFICA["metodo"], tam=30, capas=n, alcance=al, intensidad=0.09, zorder=7)

        amarillo = GRAFICA["tangente"]
        (self.tangente,) = c.plot([], [], color=amarillo, lw=1.0, ls=(0, (5, 5)), alpha=0.6, zorder=5)
        self.triangulo = Polygon([[0, 0]], closed=True, facecolor=to_rgba(amarillo, 0.08),
                                 edgecolor="none", zorder=3, visible=False)
        c.add_patch(self.triangulo)
        (self.cateto_h,) = c.plot([], [], color=amarillo, lw=1.0, ls=(0, (2, 3)), alpha=0.8, zorder=5)
        (self.cateto_v,) = c.plot([], [], color=amarillo, lw=1.0, ls=(0, (2, 3)), alpha=0.8, zorder=5)
        # clip_on: nada de la capa sale del rectángulo de los ejes (la ventana solo copia esa zona).
        estilo_txt = dict(fontsize=7.5, family=FUENTE_MONO, zorder=9, annotation_clip=True, clip_on=True)
        self.txt_h = c.annotate("", (0, 0), xytext=(0, -12), textcoords="offset points", ha="center",
                                va="top", color=amarillo, bbox=_caja(amarillo), **estilo_txt)
        self.txt_v = c.annotate("", (0, 0), xytext=(9, -4), textcoords="offset points", ha="left",
                                va="top", color=amarillo, bbox=_caja(amarillo), **estilo_txt)

        self.sondas = []     # línea del nodo a cada etapa
        self.etapas = []     # segmento con la pendiente k_i
        self.puntos_etapa = []
        self.txt_etapa = []
        n_et, a_et = H(5, 0.07)
        n_pe, al_pe = self.q.halo_puntos(3, 2.2)
        for i in range(MAX_ETAPAS):
            col = color_etapa(i)
            (sonda,) = c.plot([], [], color=col, lw=0.9, ls=(0, (1, 2.5)), alpha=0.75, zorder=5)
            self.sondas.append(sonda)
            self.etapas.append(TrazoLuminoso(c, col, lw=1.8, capas=n_et, alcance=8, intensidad=a_et, nucleo=0.4,
                                             zorder=8))
            self.puntos_etapa.append(PuntosLuminosos(c, col, tam=16, capas=n_pe, alcance=al_pe, intensidad=0.12,
                                                     nucleo=mezclar(col, "#ffffff", 0.6), zorder=8.5))
            efecto = {"path_effects": _brillo_texto(col, 3, 0.25)} if self.q.brillo >= 1 else {}
            self.txt_etapa.append(c.annotate("", (0, 0), xytext=(-9, 0), textcoords="offset points",
                                             ha="right", va="center", color=col, fontweight="bold",
                                             **efecto, **estilo_txt))

        n, a = H(8, 0.06)
        self.avance = TrazoLuminoso(c, amarillo, lw=2.4, capas=n, alcance=14, intensidad=a, nucleo=0.7,
                                    zorder=7.5)
        n, al = self.q.halo_puntos(5, 3.6)
        self.punto = PuntosLuminosos(c, GRAFICA["metodo"], tam=62, capas=n, alcance=al, intensidad=0.11,
                                     zorder=10)
        self.destino = c.scatter([], [], s=90, facecolors="none", edgecolors=amarillo, linewidths=1.2,
                                 alpha=0.8, zorder=9)
        n, a = H(5, 0.08)
        self.error = TrazoLuminoso(c, GRAFICA["error"], lw=1.8, capas=n, alcance=8, intensidad=a, nucleo=0.5,
                                   zorder=8)
        self.txt_error = c.annotate("", (0, 0), xytext=(9, 4), textcoords="offset points", ha="left",
                                    va="bottom", color=GRAFICA["error"], bbox=_caja(GRAFICA["error"]), **estilo_txt)
        self.txt_punto = c.annotate("", (0, 0), xytext=(-10, 12), textcoords="offset points", ha="right",
                                    va="bottom", color="#ffffff", fontsize=8, family=FUENTE_MONO, zorder=11,
                                    annotation_clip=True, clip_on=True, bbox=_caja(GRAFICA["metodo"], 0.8))

        self.vista_inicial = ((0.0, 1.0), (0.0, 1.0))
        self._limpiar_paso()

    # ------------------------------------------------------------------ ejes
    @property
    def ejes_base(self) -> list:
        return [self.ax, self.ax_err]

    def contiene(self, ax) -> bool:
        """¿Es ``ax`` (p. ej. ``evento.inaxes``) la gráfica principal?"""
        return ax is self.ax or ax is self.capa

    def en_franja(self, ax) -> bool:
        """¿Es ``ax`` la franja de error?"""
        return ax is self.ax_err or ax is self.capa_err

    # ------------------------------------------------------------------ carga
    def cargar(self, sim) -> None:
        """Prepara la gráfica para una simulación nueva (o ``None`` si hubo error)."""
        self.sim = sim
        self.version += 1
        self.mensaje.set_text("")
        self._verts_barras = np.empty((0, 4, 2))
        self.barras.set_verts([])
        self.err_punto.set_offsets(np.empty((0, 2)))
        if sim is None:
            for a in (self.ref, self.poligonal, self.nodos):
                a.set_visible(False)
            if self.degradado is not None:
                self.degradado.set_visible(False)
            self.ref_txt.set_text("")
            self.campo.set_segments([])
            self.err_avisos.set_offsets(np.empty((0, 2)))
            self._limpiar_paso()
            self.firma_base = (self.version,)
            return
        for a in (self.ref, self.poligonal, self.nodos):
            a.set_visible(True)

        self.xn, self.yn = sim.x_nodos, sim.y_nodos
        self.errores = sim.errores
        ok = np.isfinite(sim.y_ref)
        self.ref.set_data(sim.x_ref[ok], sim.y_ref[ok])
        if ok.sum() > 1:
            self.ref_txt.xy = (sim.x_ref[ok][-1], sim.y_ref[ok][-1])
            self.ref_txt.set_text(sim.etiqueta_referencia)
        else:
            self.ref_txt.set_text("")
        self.vista_inicial = self._vista_inicial(sim)
        self.aplicar_vista(*self.vista_inicial)
        self.ax_err.set_xlim(*self.vista_inicial[0])

        # Franja de error: rectángulos precalculados; se muestran conforme se avanza.
        e = np.where(np.isfinite(self.errores), self.errores, 0.0)
        if len(self.xn) > 1:
            medio = np.diff(self.xn) * 0.3
            x0, x1 = self.xn[1:] - medio, self.xn[1:] + medio
            alto = e[1:]
            self._verts_barras = np.stack([np.column_stack([x0, np.zeros_like(x0)]), np.column_stack([x0, alto]),
                                           np.column_stack([x1, alto]), np.column_stack([x1, np.zeros_like(x1)])],
                                          axis=1)
        tope = float(np.nanmax(e)) if e.size and np.nanmax(e) > 0 else 1.0
        self.ax_err.set_ylim(0, tope * 1.3)
        inestables = [i + 1 for i in sim.pasos_inestables if i + 1 < len(self.xn)]
        self.err_avisos.set_offsets(np.column_stack([self.xn[inestables], np.full(len(inestables), tope * 1.18)])
                                    if inestables else np.empty((0, 2)))
        self.firma_base = ()

    def mostrar_error(self, mensaje: str) -> None:
        self.cargar(None)
        self.mensaje.set_text(mensaje)
        self.firma_base = (self.version, mensaje)

    def _vista_inicial(self, sim):
        x0, x1 = sim.x0, sim.x_final
        dx = (x1 - x0) * 0.04
        ys = np.concatenate([self.yn, sim.y_ref])
        ys = ys[np.isfinite(ys)]
        lo, hi = float(ys.min()), float(ys.max())
        if ys.size > 8:
            q1, q9 = np.percentile(ys, [5, 95])
            nucleo = q9 - q1
            if nucleo > 0 and hi - lo > 12 * nucleo:      # algo explotó: no dejar que aplaste la vista
                lo, hi = max(lo, q1 - 1.5 * nucleo), min(hi, q9 + 1.5 * nucleo)
        if hi - lo < 1e-9:
            lo, hi = lo - 1, hi + 1
        dy = (hi - lo) * 0.1
        return (x0 - dx, x1 + dx), (lo - dy, hi + dy)

    # ------------------------------------------------------------------ vista
    def aplicar_vista(self, xlim, ylim) -> None:
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self.actualizar_campo()

    def actualizar_campo(self) -> None:
        if self.sim is None:
            return
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        caja = self.ax.get_window_extent()
        W, H = max(caja.width, 1), max(caja.height, 1)
        nx = 27
        ny = max(6, int(round(nx * H / W)))
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y0, y1, ny + 1)
        xs, ys = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
        X, Y = np.meshgrid(xs, ys)
        S = np.broadcast_to(np.asarray(self.sim.ecuacion(X, Y), dtype=float), X.shape)
        sx, sy = W / (x1 - x0), H / (y1 - y0)          # píxeles por unidad
        u, v = np.ones_like(S) * sx, S * sy
        norma = np.hypot(u, v)
        largo = 0.36 * min(W / nx, H / ny)
        with np.errstate(all="ignore"):
            dx = u / norma * largo / sx
            dy = v / norma * largo / sy
        ok = np.isfinite(dx) & np.isfinite(dy)
        seg = np.stack([np.stack([X - dx, Y - dy], -1), np.stack([X + dx, Y + dy], -1)], axis=-2)[ok]
        angulo = np.arctan2(v, u)[ok] / (np.pi / 2)        # -1 (bajando) ... 1 (subiendo)
        colores = _CAMPO((angulo + 1) / 2)
        colores[:, 3] = 0.55
        self.campo.set_segments(seg)
        self.campo.set_color(colores)

    # ------------------------------------------------------------------ paso a paso
    def _limpiar_paso(self) -> None:
        self.tangente.set_data([], [])
        self.triangulo.set_visible(False)
        self.cateto_h.set_data([], [])
        self.cateto_v.set_data([], [])
        for t in (self.txt_h, self.txt_v, self.txt_error, self.txt_punto):
            t.set_text("")
        for i in range(MAX_ETAPAS):     # ocultar (no solo vaciar) ahorra tiempo de dibujo
            self.sondas[i].set_visible(False)
            self.etapas[i].set_visible(False)
            self.puntos_etapa[i].set_visible(False)
            self.txt_etapa[i].set_text("")
        self.avance.set_data([], [])
        self.punto.set_offsets(np.empty((0, 2)))
        self.destino.set_offsets(np.empty((0, 2)))
        self.error.set_data([], [])

    def _hay_espacio_a_la_derecha(self, x: float) -> bool:
        x0, x1 = self.ax.get_xlim()
        return x < x0 + 0.8 * (x1 - x0)

    def punto_actual(self, revelados: int, seleccion: int, progreso: float):
        """Coordenadas del punto que hay que seguir con la cámara."""
        if self.sim is None or not self.sim.pasos:
            return (self.sim.x0, self.sim.y0) if self.sim else (0.0, 0.0)
        paso = self.sim.pasos[seleccion]
        p = progreso if seleccion == revelados - 1 else 1.0
        return paso.t + p * paso.h, float(paso.y) + p * float(paso.y_siguiente - paso.y)

    def _actualizar_franja(self, nodos_vis: int, activa: int | None) -> None:
        """Barras de error reveladas, la seleccionada encendida (solo si algo cambió)."""
        firma = (self.version, nodos_vis, activa)
        if firma == self.firma_base:
            return
        self.firma_base = firma
        n = max(nodos_vis - 1, 0)
        self.barras.set_verts(self._verts_barras[:n])
        colores = np.tile(to_rgba(GRAFICA["error"], 0.28), (n, 1))
        if activa is not None and activa < n:
            colores[activa] = to_rgba(GRAFICA["error"], 0.85)
        self.barras.set_facecolors(colores)
        if activa is not None and activa + 1 < len(self.errores) and np.isfinite(self.errores[activa + 1]):
            self.err_punto.set_offsets([[self.xn[activa + 1], self.errores[activa + 1]]])
        else:
            self.err_punto.set_offsets(np.empty((0, 2)))

    def mostrar(self, revelados: int, seleccion: int, progreso: float = 1.0, pulso: float = 1.0) -> None:
        """Dibuja el estado: ``revelados`` pasos visibles, el paso ``seleccion``
        resaltado y el último paso avanzado una fracción ``progreso``."""
        self._limpiar_paso()
        sim = self.sim
        if sim is None:
            return
        n = len(sim.pasos)
        revelados = min(revelados, n)
        xn, yn = self.xn, self.yn

        # Poligonal hasta el punto que avanza.
        if revelados == 0:
            xs, ys = xn[:1], yn[:1]
            nodos_vis = 1
        else:
            j = revelados - 1
            x_mov = xn[j] + progreso * (xn[j + 1] - xn[j])
            y_mov = yn[j] + progreso * (yn[j + 1] - yn[j])
            xs = np.r_[xn[: j + 1], x_mov]
            ys = np.r_[yn[: j + 1], y_mov]
            nodos_vis = j + 2 if progreso >= 1 else j + 1
        self.poligonal.set_data(xs, ys)
        if self.degradado is not None:
            self.degradado.set_data(xs, ys, self.ax.get_ylim()[0])
        self.nodos.set_visible(nodos_vis <= MAX_NODOS_CON_MARCA)
        self.nodos.set_offsets(np.column_stack([xn[:nodos_vis], yn[:nodos_vis]]))

        activa = seleccion if revelados and (seleccion < revelados - 1 or progreso >= 1) else None
        self._actualizar_franja(nodos_vis, activa)

        if revelados == 0:
            self.punto.set_offsets([[xn[0], yn[0]]])
            self.txt_punto.xy = (xn[0], yn[0])
            self.txt_punto.set_text(f"P₀ ({num(xn[0], 3)}, {num(yn[0], 3)})")
            return

        paso = sim.pasos[seleccion]
        p = progreso if seleccion == revelados - 1 else 1.0
        xa, ya = paso.t, float(paso.y)
        xb, yb = paso.t_siguiente, float(paso.y_siguiente)
        vx0, vx1 = self.ax.get_xlim()

        # Recta tangente (pendiente del primer paso de evaluación) a lo ancho de la vista.
        if paso.etapas:
            k1 = float(paso.etapas[0].pendiente)
            self.tangente.set_data([vx0, vx1], [ya + k1 * (vx0 - xa), ya + k1 * (vx1 - xa)])

        # Triángulo: cateto horizontal h, cateto vertical Δy = h·m.
        self.triangulo.set_xy([[xa, ya], [xb, ya], [xb, yb]])
        self.triangulo.set_visible(True)
        self.cateto_h.set_data([xa, xb], [ya, ya])
        self.cateto_v.set_data([xb, xb], [ya, yb])
        self.txt_h.xy = ((xa + xb) / 2, ya)
        self.txt_h.set_text(f"h = {num(paso.h)}")
        self.txt_h.set_va("top" if yb >= ya else "bottom")
        self.txt_h.xyann = (0, -12 if yb >= ya else 12)
        derecha = self._hay_espacio_a_la_derecha(xb)
        self.txt_v.xy = (xb, (ya + yb) / 2)
        self.txt_v.set_va("top" if yb >= ya else "bottom")
        self.txt_v.set_ha("left" if derecha else "right")
        self.txt_v.xyann = (9 if derecha else -9, -4 if yb >= ya else 4)
        nombre_m = "k₁" if len(paso.etapas) == 1 else "m"
        self.txt_v.set_text(f"Δy = h·{nombre_m} = {num(yb - ya)}")

        # Etapas: dónde evaluó f el método y con qué pendiente.
        cuarto = paso.h * 0.28
        for i, etapa in enumerate(paso.etapas[:MAX_ETAPAS]):
            te, ye, ke = etapa.t, float(etapa.y), float(etapa.pendiente)
            for a in (self.sondas[i], self.etapas[i], self.puntos_etapa[i]):
                a.set_visible(True)
            self.sondas[i].set_data([xa, te], [ya, ye])
            self.etapas[i].set_data([te - cuarto, te + cuarto], [ye - ke * cuarto, ye + ke * cuarto])
            self.etapas[i].set_brillo(pulso)
            self.puntos_etapa[i].set_offsets([[te, ye]])
            self.txt_etapa[i].xy = (te, ye)
            self.txt_etapa[i].set_text(etiqueta_sub(etapa.etiqueta))

        # El punto que avanza sobre la recta del paso.
        xp, yp = xa + p * (xb - xa), ya + p * (yb - ya)
        self.avance.set_data([xa, xp], [ya, yp])
        self.avance.set_brillo(pulso)
        self.punto.set_offsets([[xp, yp]])
        self.punto.set_brillo(pulso)
        if p < 1:
            self.destino.set_offsets([[xb, yb]])

        k = seleccion + (1 if p >= 1 else 0)
        izquierda = xp > vx0 + 0.22 * (vx1 - vx0)
        self.txt_punto.set_ha("right" if izquierda else "left")
        self.txt_punto.xyann = (-10 if izquierda else 10, 12)
        self.txt_punto.xy = (xp, yp)
        self.txt_punto.set_text(f"P{sub(k)} ({num(xp, 3)}, {num(yp, 3)})")

        # Error contra la referencia en el nodo de llegada.
        if p >= 1:
            yr = float(sim.referencia_en(xb))
            if np.isfinite(yr):
                self.error.set_data([xb, xb], [yb, yr])
                # La etiqueta del error va del lado contrario a la de Δy para no encimarse.
                arriba = yr >= yb
                self.txt_error.xy = (xb, max(yr, yb) if arriba else min(yr, yb))
                self.txt_error.set_va("bottom" if arriba else "top")
                self.txt_error.set_ha("left" if derecha else "right")
                self.txt_error.xyann = (9 if derecha else -9, 4 if arriba else -4)
                self.txt_error.set_text(f"|e{sub(seleccion + 1)}| = {abs(yr - yb):.2e}")
