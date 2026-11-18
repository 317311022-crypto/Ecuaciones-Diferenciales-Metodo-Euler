"""Efectos visuales: luminiscencia (glow) y fondo con parallax.

Luminiscencia
    Matplotlib no tiene "brillo", pero se imita bien apilando copias del mismo
    trazo, cada vez más anchas y más transparentes, debajo del trazo principal
    (y un núcleo casi blanco encima, como un tubo de neón).

Parallax
    El fondo de la gráfica son varias capas (polvo de estrellas, orbes
    desenfocados, una retícula de puntos) dibujadas en unos ejes propios, detrás
    de los ejes de datos que son transparentes. Cada capa tiene una
    *profundidad* entre 0 (infinitamente lejos) y 1 (el plano de los datos):

    * al mover la cámara (zoom, arrastrar, modo "seguir") una capa se desplaza
      ``profundidad`` veces lo que se desplazan los datos: las lejanas casi no se
      mueven y las cercanas casi acompañan a los datos;
    * al mover el ratón la vista "orbita" alrededor del plano de los datos: lo que
      está más al fondo se desplaza más.

    Las capas se repiten de forma periódica, así que nunca se ve un borde.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon

from .tema import UI


def mezclar(color, otro="#ffffff", t: float = 0.5):
    """Interpola dos colores; ``t=0`` da ``color`` y ``t=1`` da ``otro``."""
    a, b = np.array(to_rgba(color)), np.array(to_rgba(otro))
    return tuple(a + (b - a) * t)


# --------------------------------------------------------------------------- glow
class TrazoLuminoso:
    """Una línea con halo luminoso. Se usa como un ``Line2D``: ``set_data``.

    Las copias anchas y translúcidas del halo van todas en UNA ``LineCollection``
    (una sola llamada de dibujo en lugar de una por capa)."""

    def __init__(self, ax, color, lw: float = 2.0, capas: int = 8, alcance: float = 12.0,
                 intensidad: float = 0.045, nucleo: float = 0.55, zorder: float = 5,
                 ls="-", alpha: float = 1.0):
        self.intensidad = intensidad
        self.alpha = alpha
        self._n = capas
        anchos = [lw + alcance * (i + 1) / capas for i in range(capas)]
        self.halo = LineCollection([], linewidths=anchos, colors=[color], alpha=intensidad, capstyle="round",
                                   joinstyle="round", zorder=zorder - 0.02)
        ax.add_collection(self.halo, autolim=False)
        (self.linea,) = ax.plot([], [], color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
                                solid_capstyle="round", solid_joinstyle="round")
        self.nucleo = None
        if nucleo:
            (self.nucleo,) = ax.plot([], [], color=mezclar(color, "#ffffff", nucleo), lw=max(lw * 0.38, 0.6),
                                     ls=ls, alpha=alpha, zorder=zorder + 0.01,
                                     solid_capstyle="round", solid_joinstyle="round")

    @property
    def artistas(self):
        return [self.halo, self.linea] + ([self.nucleo] if self.nucleo is not None else [])

    def set_data(self, x, y) -> None:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.halo.set_segments([np.column_stack([x, y])] * self._n if len(x) else [])
        self.linea.set_data(x, y)
        if self.nucleo is not None:
            self.nucleo.set_data(x, y)

    def set_visible(self, visible: bool) -> None:
        for a in self.artistas:
            a.set_visible(visible)

    def set_brillo(self, factor: float) -> None:
        """Escala la intensidad del halo (para pulsar)."""
        self.halo.set_alpha(min(1.0, self.intensidad * factor))

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.linea.set_alpha(alpha)
        if self.nucleo is not None:
            self.nucleo.set_alpha(alpha)
        self.halo.set_alpha(self.intensidad * alpha)


class PuntosLuminosos:
    """Marcadores con halo: un núcleo claro sobre un disco de color y aros difusos.

    Todo (aros, disco y núcleo de todos los puntos) va en UNA colección: primero
    los aros, luego los discos y al final los núcleos, igual que si fueran
    colecciones separadas pero con una sola llamada de dibujo."""

    def __init__(self, ax, color, tam: float = 40, capas: int = 5, alcance: float = 3.2,
                 intensidad: float = 0.09, nucleo="#ffffff", zorder: float = 7):
        self.intensidad = intensidad
        self._capas = capas
        self._tams = np.array([tam * (1 + alcance * (i + 1) / capas) ** 2 for i in range(capas)]
                              + [tam, tam * 0.32])
        self._colores = np.array([to_rgba(color, intensidad)] * capas + [to_rgba(color, 1.0), to_rgba(nucleo, 1.0)])
        self._factor = 1.0
        self._xy = np.empty((0, 2))
        self.col = ax.scatter([], [], s=[], linewidths=0, zorder=zorder)

    @property
    def artistas(self):
        return [self.col]

    def _actualizar(self) -> None:
        n, k = len(self._xy), len(self._tams)
        self.col.set_offsets(np.tile(self._xy, (k, 1)) if n else np.empty((0, 2)))
        self.col.set_sizes(np.repeat(self._tams, n))
        colores = self._colores.copy()
        colores[: self._capas, 3] = min(1.0, self.intensidad * self._factor)
        self.col.set_facecolors(np.repeat(colores, n, axis=0))

    def set_offsets(self, xy) -> None:
        self._xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        self._actualizar()

    def set_visible(self, visible: bool) -> None:
        self.col.set_visible(visible)

    def set_brillo(self, factor: float) -> None:
        if factor != self._factor:
            self._factor = factor
            self._actualizar()


class DegradadoBajoCurva:
    """Resplandor bajo una curva: varias franjas translúcidas apiladas que se
    van superponiendo cerca de la curva, lo que da un degradado vertical."""

    def __init__(self, ax, color, alpha: float = 0.05, franjas=(0.025, 0.06, 0.11, 0.19), zorder: float = 3):
        self.ax = ax
        self.franjas = franjas
        self.polis = [ax.add_patch(Polygon([[0, 0]], closed=True, facecolor=color, edgecolor="none",
                                           alpha=alpha, zorder=zorder, visible=False)) for _ in franjas]

    def set_data(self, x, y, piso: float) -> None:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(x) < 2 or not np.all(np.isfinite(y)):
            self.set_visible(False)
            return
        y0, y1 = self.ax.get_ylim()
        alto = abs(y1 - y0)
        for fr, poli in zip(self.franjas, self.polis):
            abajo = np.maximum(y - fr * alto, piso)
            poli.set_xy(np.column_stack([np.r_[x, x[::-1]], np.r_[y, abajo[::-1]]]))
            poli.set_visible(True)

    def set_visible(self, visible: bool) -> None:
        for p in self.polis:
            p.set_visible(visible)


def resplandor(ax, capas: int = 6, alcance: float = 8.0, intensidad: float = 0.05) -> None:
    """Agrega halo luminoso a todas las líneas sólidas de unos ejes (para gráficas estáticas)."""
    for linea in list(ax.get_lines()):
        if not linea.get_visible() or linea.get_linestyle() in ("None", "", " "):
            continue
        x, y = linea.get_data()
        lw = linea.get_linewidth()
        for i in range(capas):
            ax.plot(x, y, color=linea.get_color(), lw=lw + alcance * (i + 1) / capas, alpha=intensidad,
                    zorder=linea.get_zorder() - 0.1, solid_capstyle="round", label="_halo")


# --------------------------------------------------------------------------- parallax
class _Capa:
    """Capa de puntos repetida periódicamente. ``base`` vive en [0, W)²."""

    margen = 0.12

    def __init__(self, profundidad: float):
        self.profundidad = profundidad

    def _envolver(self, base, dx, dy):
        w = 1 + 2 * self.margen
        x = np.mod(base[:, 0] + dx, w) - self.margen
        y = np.mod(base[:, 1] + dy, w) - self.margen
        return np.column_stack([x, y])


class CapaEstrellas(_Capa):
    def __init__(self, ax, rng, n, profundidad, tam, alpha, colores, zorder):
        super().__init__(profundidad)
        w = 1 + 2 * self.margen
        self.base = rng.uniform(0, w, size=(n, 2))
        self.alpha_base = rng.uniform(*alpha, size=n)
        self.fase = rng.uniform(0, 2 * np.pi, size=n)
        self.vel = rng.uniform(0.6, 2.2, size=n)
        self.rgba = np.array([to_rgba(colores[i % len(colores)]) for i in rng.integers(0, len(colores), n)])
        tamanos = rng.uniform(*tam, size=n) ** 2
        self.puntos = ax.scatter(*self.base.T, s=tamanos, c=self.rgba, linewidths=0, zorder=zorder)
        self._colores(0.0)

    def _colores(self, t):
        rgba = self.rgba.copy()
        rgba[:, 3] = self.alpha_base * (0.7 + 0.3 * np.sin(self.vel * t + self.fase))
        self.puntos.set_facecolors(rgba)

    def desplazar(self, dx, dy, t=None):
        self.puntos.set_offsets(self._envolver(self.base, dx, dy))
        if t is not None:
            self._colores(t)


class CapaOrbes(_Capa):
    """Círculos grandes y muy transparentes: el 'bokeh' de una cámara desenfocada."""

    margen = 0.3

    def __init__(self, ax, rng, n, profundidad, colores, zorder):
        super().__init__(profundidad)
        w = 1 + 2 * self.margen
        self.base = rng.uniform(0, w, size=(n, 2))
        radios = rng.uniform(18, 70, size=n)
        cols = [colores[i % len(colores)] for i in range(n)]
        self.aros = []
        for factor, alpha in ((1.0, 0.035), (0.72, 0.035), (0.45, 0.04)):
            self.aros.append(ax.scatter(*self.base.T, s=(radios * factor) ** 2, c=cols, alpha=alpha,
                                        linewidths=0, zorder=zorder))

    def desplazar(self, dx, dy, t=None):
        xy = self._envolver(self.base, dx, dy)
        for a in self.aros:
            a.set_offsets(xy)


class CapaReticula(_Capa):
    """Retícula de puntos (como en un editor de diseño), alineada a una malla regular."""

    def __init__(self, ax, profundidad, paso, color, zorder):
        super().__init__(profundidad)
        self.paso = paso
        g = np.arange(-paso, 1 + 2 * paso, paso)
        X, Y = np.meshgrid(g, g)
        self.base = np.column_stack([X.ravel(), Y.ravel()])
        self.puntos = ax.scatter(*self.base.T, s=1.6, color=color, linewidths=0, zorder=zorder)

    def desplazar(self, dx, dy, t=None):
        self.puntos.set_offsets(self.base + [np.mod(dx, self.paso), np.mod(dy, self.paso)])


class FondoParallax:
    """Fondo de la gráfica.

    ``ax_fijo`` guarda una sola imagen (degradado, nebulosa y viñeta) y ``ax``
    las capas decorativas (estrellas, orbes, retícula). Con ``parallax=True``
    las capas se desplazan con la cámara y el ratón, lo que obliga a
    redibujarlas en cada movimiento; con ``parallax=False`` quedan fijas y se
    dibujan una sola vez junto con el resto de la parte estática de la ventana.
    """

    SENSIBILIDAD_RATON = 0.07

    def __init__(self, fig, rect, parallax: bool = True, decoracion: bool = True, centelleo: bool = True,
                 semilla: int = 7, zorder: float = -5):
        self.parallax = parallax and decoracion
        self.centelleo = centelleo and self.parallax
        self.ax_fijo = fig.add_axes(rect, zorder=zorder - 0.5)
        self.ax = fig.add_axes(rect, zorder=zorder)
        for ax in (self.ax_fijo, self.ax):
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_navigate(False)
        self.ax.patch.set_visible(False)
        rng = np.random.default_rng(semilla)

        # uint8 a baja resolución + bilineal: unas 3-4 veces más rápido de dibujar que flotantes.
        self.ax_fijo.imshow((self._imagen_fija() * 255).astype(np.uint8), extent=(0, 1, 0, 1), origin="lower",
                            aspect="auto", interpolation="bilinear", zorder=0)

        estrellas = ["#ffffff", "#9fdcff", "#d9c2ff", "#ffd6f5"]
        self.capas = []
        if decoracion:
            self.capas = [
                CapaEstrellas(self.ax, rng, 260, 0.10, (0.4, 1.6), (0.15, 0.55), estrellas, zorder=2),
                CapaOrbes(self.ax, rng, 11, 0.25, ["#ff2bd6", "#00e5ff", "#8a5cff", "#00ffc8", "#3d7bff"], zorder=3),
                CapaEstrellas(self.ax, rng, 60, 0.42, (1.2, 2.8), (0.35, 0.9), estrellas, zorder=4),
                CapaReticula(self.ax, 0.7, 0.045, "#1d2a52", zorder=5),
            ]
        self.firma: tuple = ()
        for capa in self.capas:
            capa.desplazar(0.0, 0.0)

    @staticmethod
    def _imagen_fija(n: int = 180) -> np.ndarray:
        """Degradado radial azul noche + manchas de nebulosa + viñeta, en RGB."""
        yy, xx = np.mgrid[0:1:n * 1j, 0:1:int(n * 1.6) * 1j]
        centro, orilla = np.array(to_rgba("#0c1430")[:3]), np.array(to_rgba(UI["fondo"])[:3])
        r = np.hypot((xx - 0.55) * 1.2, (yy - 0.6))
        t = np.clip(r / 0.85, 0, 1)[..., None] ** 1.3
        img = centro * (1 - t) + orilla * t
        for cx, cy, s, col, a in [(0.25, 0.75, 0.22, "#6a00ff", 0.22), (0.8, 0.3, 0.25, "#00b3ff", 0.15),
                                  (0.62, 0.85, 0.18, "#ff2bd6", 0.12), (0.15, 0.2, 0.2, "#00ffc8", 0.07)]:
            g = (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s**2)) * a)[..., None]
            img = img * (1 - g) + np.array(to_rgba(col)[:3]) * g
        r = np.hypot(xx - 0.5, yy - 0.5)
        v = (np.clip((r - 0.32) / 0.42, 0, 1) ** 1.7 * 0.8)[..., None]
        img = img * (1 - v) + orilla * v
        return np.clip(img, 0, 1)

    @property
    def ejes_moviles(self) -> list:
        """Ejes que hay que redibujar cuando la cámara o el ratón se mueven."""
        return [self.ax] if self.parallax else []

    def actualizar(self, camara, raton, t: float | None = None) -> None:
        """``camara``: desplazamiento de la vista de datos (en anchos de la vista
        inicial). ``raton``: posición del ratón respecto al centro, en [-0.5, 0.5].
        Sin parallax no hace nada."""
        if not self.parallax:
            return
        cx, cy = camara
        rx, ry = raton
        if not self.centelleo:
            t = None
        k = self.SENSIBILIDAD_RATON
        for capa in self.capas:
            p = capa.profundidad
            capa.desplazar(-p * cx + (1 - p) * k * rx, -p * cy + (1 - p) * k * ry, t)
        # Lo que hay que comparar para saber si el fondo cambió desde el último dibujo.
        self.firma = (round(float(cx), 5), round(float(cy), 5), round(float(rx), 4), round(float(ry), 4),
                      None if t is None else round(t, 2))
