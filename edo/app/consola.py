"""Panel de registro con resaltado de sintaxis, al estilo de un editor de código.

Cada renglón es una lista de *tokens* ``(texto, estilo)``. Para pintar un
renglón de varios colores con matplotlib (que solo da un color por ``Text``)
se usa un truco con la fuente monoespaciada: por cada estilo presente se
dibuja el renglón completo en el que los caracteres de *otros* estilos se
cambian por espacios. Todos los textos de un renglón comparten posición y
línea base, así que los colores quedan perfectamente alineados.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from matplotlib.patches import Rectangle

from .tema import CONSOLA, FUENTE_MONO, GRAFICA, UI

ANCHO_CARACTER = 0.6021   # avance de DejaVu Sans Mono, en ems


@dataclass
class Linea:
    tokens: list[tuple[str, str]] = field(default_factory=list)
    paso: Optional[int] = None     # paso del método al que pertenece el renglón
    enfasis: bool = False          # renglón con brillo (encabezados de paso)

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
        self._pool: dict[tuple[int, str], object] = {}
        self._usados: set[tuple[int, str]] = set()

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
        for (r, _), txt in self._pool.items():
            txt.set_y(self._y_base(r))
        self.limitar()

    def _y_base(self, fila: int) -> float:
        return 1 - self.margen_sup - (fila + 0.78) * self.fila

    def _texto(self, fila: int, estilo: str):
        clave = (fila, estilo)
        txt = self._pool.get(clave)
        if txt is None:
            txt = self.ax.text(self.x_texto, self._y_base(fila), "", transform=self.ax.transAxes,
                               family=FUENTE_MONO, fontsize=self.tam, color=CONSOLA.get(estilo, CONSOLA["texto"]),
                               ha="left", va="baseline", clip_on=True, parse_math=False, zorder=2)
            self._pool[clave] = txt
        return txt

    # ------------------------------------------------------------------ contenido
    def establecer(self, lineas: list[Linea]) -> None:
        self.lineas = list(lineas)
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
        usados: set[tuple[int, str]] = set()
        visibles_sel = []
        for r in range(self.filas):
            i = self.inicio + r
            if i >= len(self.lineas):
                break
            linea = self.lineas[i]
            activa = self.seleccion is not None and linea.paso == self.seleccion
            if activa:
                visibles_sel.append(r)
            tokens = [(f"{i + 1:>4}  ", "gutter_activo" if activa else "gutter")] + linea.tokens
            # Recorta a lo que cabe en el panel.
            total, recortados = 0, []
            for texto, estilo in tokens:
                if total >= self.columnas:
                    break
                texto = texto[: self.columnas - total]
                recortados.append((texto, estilo))
                total += len(texto)
            estilos = {e for t, e in recortados if t.strip()}
            for estilo in estilos:
                cadena = "".join(t if e == estilo else " " * len(t) for t, e in recortados).rstrip()
                txt = self._texto(r, estilo)
                txt.set_text(cadena)
                txt.set_fontweight("bold" if linea.enfasis and estilo not in ("gutter", "gutter_activo") else "normal")
                usados.add((r, estilo))
        for clave in self._usados - usados:
            self._pool[clave].set_text("")
        self._usados = usados

        # Bloque seleccionado.
        if visibles_sel:
            r0, r1 = min(visibles_sel), max(visibles_sel)
            y_top = 1 - self.margen_sup - r0 * self.fila + self.fila * 0.12
            alto = (r1 - r0 + 1) * self.fila
            for p in (self.resaltado, self.acento, self.acento_halo):
                p.set_y(y_top - alto)
                p.set_height(alto)
                p.set_visible(True)
        else:
            for p in (self.resaltado, self.acento, self.acento_halo):
                p.set_visible(False)

        # Barra de desplazamiento.
        n = len(self.lineas)
        if n > self.filas:
            alto = max(self.filas / n, 0.04)
            y_top = 1 - (self.inicio / n) * 1
            self.pulgar.set_height(alto)
            self.pulgar.set_y(max(0.0, y_top - alto))
            self.pista.set_visible(True)
            self.pulgar.set_visible(True)
        else:
            self.pista.set_visible(False)
            self.pulgar.set_visible(False)
