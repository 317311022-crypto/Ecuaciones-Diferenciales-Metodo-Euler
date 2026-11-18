"""Panel de registro con resaltado de sintaxis, al estilo de un editor de código.

Cada renglón es una lista de *tokens* ``(texto, estilo)``.

Rendimiento
-----------
Matplotlib solo da un color por ``Text``, así que un registro de colores
necesitaría cientos de objetos ``Text``, y cada uno vuelve a rasterizar sus
letras con FreeType en cada dibujo (es lo más caro). Aquí el panel se dibuja
como **una sola imagen**:

1. Cada carácter se rasteriza una sola vez con FreeType (la misma biblioteca y
   los mismos ajustes que usa matplotlib) y se guarda en un *atlas*.
2. Cada renglón se arma pegando esos mapas de bits en una cuadrícula
   monoespaciada y pintando cada columna del color de su token. El renglón
   armado también se guarda en caché: al revelar un paso o desplazar el
   registro solo se arman los renglones nuevos.
3. El panel completo se envía al renderizador con ``draw_image``.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from matplotlib.artist import Artist
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont, get_font
from matplotlib.patches import Rectangle

from .tema import CONSOLA, FUENTE_MONO, GRAFICA, UI

try:   # mismo modo de "hinting" que el texto normal de matplotlib
    from matplotlib.backends.backend_agg import get_hinting_flag
except ImportError:  # pragma: no cover
    def get_hinting_flag():
        return 0

ANCHO_CARACTER = 0.6021   # avance de DejaVu Sans Mono, en ems
BASE_RENGLON = 0.78       # línea base dentro de la franja de cada renglón (fracción desde arriba)
_ESTILOS_SIN_NEGRITA = ("gutter", "gutter_activo")


@dataclass
class Linea:
    tokens: list[tuple[str, str]] = field(default_factory=list)
    paso: Optional[int] = None     # paso del método al que pertenece el renglón
    enfasis: bool = False          # renglón destacado (encabezados de paso): en negritas

    @property
    def texto(self) -> str:
        return "".join(t for t, _ in self.tokens)


def L(*tokens, paso: Optional[int] = None, enfasis: bool = False) -> Linea:
    """Atajo: ``L(("paso", "clave"), (" 3", "numero"))``. Un str suelto usa estilo 'texto'."""
    norm = [(t, "texto") if isinstance(t, str) else (t[0], t[1]) for t in tokens]
    return Linea(norm, paso, enfasis)


_SUB_A_DIGITO = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_LEXER = re.compile(
    r"(?P<num>\d+\.\d+(?:e[+-]?\d+)?|\d+(?:e[+-]?\d+)?|nan|∞)"
    r"|(?P<nom>[A-Za-zÁÉÍÓÚáéíóúñ_]+[₀-₉ₙ₊₋]*)"
    r"|(?P<op>[=+\-*/·×()^,≈|])"
    r"|(?P<otro>\s+|.)"
)


def colorear(texto: str) -> list[tuple[str, str]]:
    """Tokeniza una fórmula en texto plano y asigna estilos (números, variables, k₁...)."""
    tokens: list[tuple[str, str]] = []
    for m in _LEXER.finditer(texto):
        tipo, valor = m.lastgroup, m.group()
        if tipo == "num":
            estilo = "numero"
        elif tipo == "nom":
            siguiente = texto[m.end():m.end() + 1]
            simple = valor.translate(_SUB_A_DIGITO)
            if re.fullmatch(r"k\d", simple):
                estilo = simple
            elif siguiente == "(" and len(valor) > 1 or valor == "f":
                estilo = "funcion"
            else:
                estilo = "variable"
        elif tipo == "op":
            estilo = "operador"
        else:
            estilo = "texto"
        if tokens and tokens[-1][1] == estilo:
            tokens[-1] = (tokens[-1][0] + valor, estilo)
        else:
            tokens.append((valor, estilo))
    return tokens


# ----------------------------------------------------------------------------------- rasterizado
class _Atlas:
    """Mapa de bits de cada carácter, rasterizado una vez con FreeType."""

    def __init__(self):
        self._glifos: dict = {}

    def glifo(self, ch: str, negrita: bool, tam: float, dpi: float):
        clave = (ch, negrita, tam, round(dpi, 3))
        g = self._glifos.get(clave)
        if g is None:
            fp = FontProperties(family=FUENTE_MONO, size=tam, weight="bold" if negrita else "normal")
            fuente = get_font(findfont(fp))
            fuente.clear()
            fuente.set_size(tam, dpi)
            fuente.set_text(ch, 0, flags=get_hinting_flag())
            fuente.draw_glyphs_to_bitmap(antialiased=True)
            imagen = np.array(fuente.get_image(), dtype=np.uint8, copy=True)
            g = (imagen, fuente.get_descent() / 64.0, fuente.get_bitmap_offset()[0] / 64.0)
            self._glifos[clave] = g
        return g


_ATLAS = _Atlas()
_RGB: dict = {}


def _rgb(estilo: str) -> np.ndarray:
    if estilo not in _RGB:
        _RGB[estilo] = (np.array(to_rgb(CONSOLA.get(estilo, CONSOLA["texto"]))) * 255).round().astype(np.uint8)
    return _RGB[estilo]


class _TextoConsola(Artist):
    """Dibuja los renglones visibles de la consola como una sola imagen."""

    MAX_CACHE = 4000

    def __init__(self, consola: "Consola"):
        super().__init__()
        self.consola = consola
        self._renglones: OrderedDict = OrderedDict()
        self.set_zorder(2)

    def _renglon(self, tokens, enfasis, tam, dpi, alto, avance):
        """(alfa, colores) de un renglón: alfa de alto×ancho y un color por columna de píxeles."""
        clave = (tokens, enfasis, tam, round(dpi, 3), alto, round(avance, 5))
        hecho = self._renglones.get(clave)
        if hecho is not None:
            self._renglones.move_to_end(clave)
            return hecho
        n = sum(len(t) for t, _ in tokens)
        ancho = int(math.ceil(n * avance)) + 6
        alfa = np.zeros((alto, ancho), np.uint8)
        colores = np.zeros((ancho, 3), np.uint8)
        base = BASE_RENGLON * alto
        col = 0
        ultimo = 0
        for texto, estilo in tokens:
            rgb = _rgb(estilo)
            negrita = enfasis and estilo not in _ESTILOS_SIN_NEGRITA
            x_ini, x_fin = int(round(col * avance)), int(round((col + len(texto)) * avance))
            colores[x_ini:x_fin] = rgb
            ultimo = x_fin
            for ch in texto:
                if ch != " ":
                    img, d, xo = _ATLAS.glifo(ch, negrita, tam, dpi)
                    h, w = img.shape
                    if h and w:
                        izq = int(round(col * avance + xo))
                        arriba = int(round(base + d)) + 1 - h       # igual que el texto de Agg
                        f0, f1 = max(arriba, 0), min(arriba + h, alto)
                        c0, c1 = max(izq, 0), min(izq + w, ancho)
                        if f1 > f0 and c1 > c0:
                            zona = alfa[f0:f1, c0:c1]
                            np.maximum(zona, img[f0 - arriba:f1 - arriba, c0 - izq:c1 - izq], out=zona)
                col += 1
        if ultimo < ancho and tokens:
            colores[ultimo:] = _rgb(tokens[-1][1])     # tinta que sobresale del último carácter
        hecho = (alfa, colores)
        self._renglones[clave] = hecho
        if len(self._renglones) > self.MAX_CACHE:
            self._renglones.popitem(last=False)
        return hecho

    def draw(self, renderer):
        c = self.consola
        if not self.get_visible() or not c._visibles:
            return
        caja = c.ax.bbox
        x0, y0 = int(round(caja.x0)), int(round(caja.y0))
        W, H = int(round(caja.x1)) - x0, int(round(caja.y1)) - y0
        if W <= 0 or H <= 0:
            return
        dpi = self.figure.dpi
        alto = max(1, int(round(c.fila * H)))
        avance = c.tam * ANCHO_CARACTER * dpi / 72
        imagen = np.zeros((H, W, 4), np.uint8)
        x_texto = int(round(c.x_texto * W))
        for r, tokens, enfasis in c._visibles:
            alfa, colores = self._renglon(tokens, enfasis, c.tam, dpi, alto, avance)
            arriba = int(round((c.margen_sup + r * c.fila) * H))
            h = min(alfa.shape[0], H - arriba)
            w = min(alfa.shape[1], W - x_texto)
            if h <= 0 or w <= 0 or arriba < 0:
                continue
            zona = imagen[arriba:arriba + h, x_texto:x_texto + w]
            zona[..., :3] = colores[None, :w]
            zona[..., 3] = np.maximum(zona[..., 3], alfa[:h, :w])
        gc = renderer.new_gc()
        gc.set_clip_rectangle(caja)
        renderer.draw_image(gc, x0, y0, np.ascontiguousarray(imagen[::-1]))   # draw_image: fila 0 abajo
        gc.restore()
        self.stale = False


# ----------------------------------------------------------------------------------- consola
class Consola:
    """Registro desplazable con números de renglón, selección y barra de scroll."""

    def __init__(self, fig, rect, tam: float = 9.0, interlineado: float = 1.5):
        self.fig = fig
        self.ax = fig.add_axes(rect, zorder=1)
        self.ax.set_facecolor(UI["panel"])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_navigate(False)
        for s in self.ax.spines.values():
            s.set_visible(False)

        self.tam = tam
        self.interlineado = interlineado
        self.lineas: list[Linea] = []
        self.inicio = 0
        self.seleccion: Optional[int] = None
        self.pegado_al_final = True
        self._visibles: list = []

        t = self.ax.transAxes
        self.resaltado = Rectangle((0, 0), 1, 0, transform=t, color=UI["seleccion"], zorder=0.5, visible=False)
        self.acento_halo = Rectangle((0, 0), 0.02, 0, transform=t, color=GRAFICA["metodo"], alpha=0.18,
                                     zorder=0.55, visible=False, linewidth=0)
        self.acento = Rectangle((0, 0), 0.006, 0, transform=t, color=GRAFICA["metodo"], zorder=0.6,
                                visible=False, linewidth=0)
        self.pista = Rectangle((0.989, 0), 0.006, 1, transform=t, color=UI["borde_suave"], zorder=0.5,
                               visible=False, linewidth=0)
        self.pulgar = Rectangle((0.989, 0), 0.006, 0, transform=t, color="#2c3a63", zorder=0.6,
                                visible=False, linewidth=0)
        for p in (self.resaltado, self.acento_halo, self.acento, self.pista, self.pulgar):
            self.ax.add_patch(p)
        self.texto = _TextoConsola(self)
        self.ax.add_artist(self.texto)
        self.medir()

    # ------------------------------------------------------------------ geometría
    def medir(self) -> None:
        """Recalcula cuántos renglones caben (llamar al cambiar el tamaño de la ventana)."""
        caja = self.ax.get_window_extent()
        alto, ancho = max(caja.height, 1.0), max(caja.width, 1.0)
        fila_px = self.tam * self.interlineado * self.fig.dpi / 72
        pad = 10 * self.fig.dpi / 72
        self.fila = fila_px / alto
        self.margen_sup = pad / alto
        self.x_texto = 8 * self.fig.dpi / 72 / ancho
        self.filas = max(1, int((alto - 2 * pad) // fila_px))
        self.columnas = max(10, int((ancho - 20 * self.fig.dpi / 72) // (self.tam * ANCHO_CARACTER * self.fig.dpi / 72)))
        self.limitar()

    def _y_base(self, fila: int) -> float:
        return 1 - self.margen_sup - (fila + BASE_RENGLON) * self.fila

    # ------------------------------------------------------------------ contenido
    def establecer(self, lineas: list[Linea]) -> None:
        self.lineas = lineas if isinstance(lineas, list) else list(lineas)
        self.limitar()

    def limitar(self) -> None:
        maximo = max(0, len(self.lineas) - self.filas)
        self.inicio = min(max(self.inicio, 0), maximo)
        self.pegado_al_final = self.inicio >= maximo

    def desplazar(self, renglones: int) -> None:
        self.inicio += renglones
        self.limitar()

    def ir_al_final(self) -> None:
        self.inicio = len(self.lineas)
        self.limitar()

    def asegurar_visible(self, paso: int, holgura: int = 2) -> None:
        """Desplaza lo mínimo para que el bloque del ``paso`` quede a la vista."""
        idx = [i for i, ln in enumerate(self.lineas) if ln.paso == paso]
        if not idx:
            return
        primero, ultimo = idx[0], idx[-1]
        if ultimo + holgura >= self.inicio + self.filas:
            self.inicio = ultimo + holgura - self.filas + 1
        if primero - 1 < self.inicio:
            self.inicio = primero - 1
        self.limitar()

    def paso_en(self, y_ejes: float) -> Optional[int]:
        """Paso del renglón bajo la coordenada ``y`` (en fracción de los ejes)."""
        r = int((1 - self.margen_sup - y_ejes) // self.fila)
        i = self.inicio + r
        if 0 <= r < self.filas and 0 <= i < len(self.lineas):
            return self.lineas[i].paso
        return None

    def coordenadas(self, indice: int, columna: int):
        """Posición (fracción de los ejes) del carácter ``columna`` del renglón
        ``indice``, o ``None`` si no está a la vista."""
        r = indice - self.inicio
        if not 0 <= r < self.filas:
            return None
        ancho = self.ax.get_window_extent().width
        x = self.x_texto + (6 + columna) * self.tam * ANCHO_CARACTER * self.fig.dpi / 72 / max(ancho, 1)
        return x, self._y_base(r)

    # ------------------------------------------------------------------ dibujo
    def renderizar(self) -> None:
        """Prepara los renglones visibles, la selección y la barra de desplazamiento."""
        visibles = []
        filas_sel = []
        for r in range(self.filas):
            i = self.inicio + r
            if i >= len(self.lineas):
                break
            linea = self.lineas[i]
            activa = self.seleccion is not None and linea.paso == self.seleccion
            if activa:
                filas_sel.append(r)
            tokens = [(f"{i + 1:>4}  ", "gutter_activo" if activa else "gutter")] + linea.tokens
            total, recortados = 0, []
            for texto, estilo in tokens:      # recorta a lo que cabe en el panel
                if total >= self.columnas:
                    break
                texto = texto[: self.columnas - total]
                recortados.append((texto, estilo))
                total += len(texto)
            visibles.append((r, tuple(recortados), linea.enfasis))
        self._visibles = visibles
        self.texto.stale = True

        if filas_sel:
            r0, r1 = min(filas_sel), max(filas_sel)
            y_top = 1 - self.margen_sup - r0 * self.fila + self.fila * 0.12
            alto = (r1 - r0 + 1) * self.fila
            for p in (self.resaltado, self.acento, self.acento_halo):
                p.set_y(y_top - alto)
                p.set_height(alto)
                p.set_visible(True)
        else:
            for p in (self.resaltado, self.acento, self.acento_halo):
                p.set_visible(False)

        n = len(self.lineas)
        if n > self.filas:
            alto = max(self.filas / n, 0.04)
            y_top = 1 - self.inicio / n
            self.pulgar.set_height(alto)
            self.pulgar.set_y(max(0.0, y_top - alto))
            self.pista.set_visible(True)
            self.pulgar.set_visible(True)
        else:
            self.pista.set_visible(False)
            self.pulgar.set_visible(False)
