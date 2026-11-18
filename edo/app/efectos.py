"""Efectos visuales: luminiscencia (glow) y fondo estelar fijo.

Luminiscencia
    Matplotlib no tiene "brillo", pero se imita bien apilando copias del mismo
    trazo, cada vez más anchas y más transparentes, debajo del trazo principal
    (y un núcleo casi blanco encima, como un tubo de neón).

Fondo
    Detrás de los ejes de datos (que son transparentes) hay una imagen con un
    degradado, manchas de nebulosa y viñeta, y encima polvo de estrellas, orbes
    desenfocados y una retícula de puntos. Todo es fijo: se dibuja una sola vez
    por tamaño de ventana y después solo se copia.
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


# --------------------------------------------------------------------------- fondo
class FondoEstelar:
    """Fondo fijo de la gráfica.

    ``ax_fijo`` guarda una sola imagen (degradado, nebulosa y viñeta) y ``ax`` la
    decoración (estrellas, orbes, retícula de puntos). Nada se mueve, así que la
    ventana los dibuja una vez y después reutiliza el resultado.
    """

    ESTRELLAS = ("#ffffff", "#9fdcff", "#d9c2ff", "#ffd6f5")
    ORBES = ("#ff2bd6", "#00e5ff", "#8a5cff", "#00ffc8", "#3d7bff")

    def __init__(self, fig, rect, decoracion: bool = True, semilla: int = 7, zorder: float = -5):
        self.ax_fijo = fig.add_axes(rect, zorder=zorder - 0.5)
        self.ax = fig.add_axes(rect, zorder=zorder)
        for ax in (self.ax_fijo, self.ax):
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_navigate(False)
        self.ax.patch.set_visible(False)

        # uint8 a baja resolución + bilineal: unas 3-4 veces más rápido de dibujar que flotantes.
        self.ax_fijo.imshow((self._imagen_fija() * 255).astype(np.uint8), extent=(0, 1, 0, 1), origin="lower",
                            aspect="auto", interpolation="bilinear", zorder=0)
        if decoracion:
            self._decorar(np.random.default_rng(semilla))

    def _decorar(self, rng) -> None:
        ax = self.ax
        # Dos capas de polvo de estrellas: muchas pequeñas y tenues, pocas más grandes.
        for n, tam, alpha, z in ((260, (0.4, 1.6), (0.15, 0.55), 2), (60, (1.2, 2.8), (0.35, 0.9), 4)):
            xy = rng.uniform(-0.12, 1.12, size=(n, 2))
            rgba = np.array([to_rgba(c) for c in rng.choice(self.ESTRELLAS, n)])
            rgba[:, 3] = rng.uniform(*alpha, size=n) * (0.7 + 0.3 * np.sin(rng.uniform(0, 2 * np.pi, size=n)))
            ax.scatter(*xy.T, s=rng.uniform(*tam, size=n) ** 2, c=rgba, linewidths=0, zorder=z)
        # Orbes: círculos grandes y muy transparentes, como el "bokeh" de una cámara desenfocada.
        xy = rng.uniform(-0.3, 1.3, size=(11, 2))
        radios = rng.uniform(18, 70, size=11)
        colores = [self.ORBES[i % len(self.ORBES)] for i in range(11)]
        for factor, alpha in ((1.0, 0.035), (0.72, 0.035), (0.45, 0.04)):
            ax.scatter(*xy.T, s=(radios * factor) ** 2, c=colores, alpha=alpha, linewidths=0, zorder=3)
        # Retícula de puntos (como en un editor de diseño).
        g = np.arange(0, 1 + 1e-9, 0.045)
        X, Y = np.meshgrid(g, g)
        ax.scatter(X.ravel(), Y.ravel(), s=1.6, color="#1d2a52", linewidths=0, zorder=5)

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
