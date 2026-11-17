"""Tema oscuro de alto contraste, inspirado en un entorno de desarrollo.

Todos los colores de la app salen de aquí: para cambiar el aspecto basta con
editar estos diccionarios. Los colores de las series de la gráfica (método,
referencia, tangente, etapas, error) se eligieron para distinguirse entre sí
también con daltonismo (deuteranopia/protanopia); además cada elemento lleva
una segunda pista visual (estilo de línea, etiqueta), no solo el color.
"""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler

FUENTE_UI = "DejaVu Sans"
FUENTE_MONO = "DejaVu Sans Mono"   # incluida con matplotlib: monoespaciada garantizada

UI = {
    "fondo": "#05070d",          # ventana
    "panel": "#090c15",          # paneles
    "panel_alto": "#0d1220",     # cabeceras, barra de entradas
    "campo": "#0f1527",          # cajas de texto
    "campo_hover": "#141c33",
    "borde": "#1a2236",
    "borde_suave": "#121829",
    "borde_activo": "#00e5ff",
    "texto": "#dbe4ff",
    "texto_medio": "#8a97bd",
    "texto_tenue": "#56628a",
    "estado": "#14102b",         # barra de estado
    "acento": "#ff2bd6",
    "acento_2": "#00e5ff",
    "ok": "#5dff9b",
    "aviso": "#ffd43b",
    "error": "#ff4d4d",
    "seleccion": "#141b31",      # renglón resaltado del registro
}

GRAFICA = {
    "metodo": "#ff2bd6",         # poligonal de la solución numérica (magenta neón)
    "referencia": "#00e5ff",     # solución exacta / referencia (cian)
    "tangente": "#ffd43b",       # recta tangente del paso actual (amarillo)
    "error": "#ff4d4d",          # error en el nodo (rojo)
    "nodo": "#ffffff",
    "rejilla": "#141d33",
    "eje": "#2a3554",
    "ticks": "#7d8ab0",
    "campo_bajo": "#3b2a7a",     # campo de pendientes: pendiente negativa
    "campo_medio": "#1f4f8f",
    "campo_alto": "#00a2b8",     # pendiente positiva
}

ETAPAS = ["#ffd43b", "#b483ff", "#ff8a3d", "#5dff9b", "#7ab8ff", "#ff9ecf"]
"""Color de k₁, k₂, k₃... (iguales en el registro y en la gráfica)."""

CONSOLA = {
    "texto": "#c8d3f5",
    "tenue": "#56628a",
    "comentario": "#4f5d86",
    "clave": "#c792ea",
    "numero": "#ff9e64",
    "variable": "#82aaff",
    "funcion": "#ffc777",
    "operador": "#89ddff",
    "cadena": "#c3e88d",
    "titulo": "#7dfcff",
    "metodo": GRAFICA["metodo"],
    "ok": "#5dff9b",
    "info": "#7ab8ff",
    "aviso": "#ffd43b",
    "error": "#ff4d4d",
    "gutter": "#323d5e",
    "gutter_activo": "#9aa7d6",
    **{f"k{i + 1}": c for i, c in enumerate(ETAPAS)},
}
"""Estilos de token del registro (cada token se pinta con uno de estos)."""


def color_etapa(i: int) -> str:
    return ETAPAS[i % len(ETAPAS)]


def aplicar_tema(app: bool = False) -> None:
    """Ajusta ``rcParams`` para el tema oscuro.

    Con ``app=True`` además quita la barra de herramientas y los atajos de
    teclado por defecto de matplotlib (la app define los suyos).
    """
    mpl.rcParams.update({
        "figure.facecolor": UI["fondo"],
        "savefig.facecolor": UI["fondo"],
        "axes.facecolor": UI["panel"],
        "axes.edgecolor": GRAFICA["eje"],
        "axes.labelcolor": UI["texto_medio"],
        "axes.titlecolor": UI["texto"],
        "text.color": UI["texto"],
        "xtick.color": GRAFICA["ticks"],
        "ytick.color": GRAFICA["ticks"],
        "grid.color": GRAFICA["rejilla"],
        "font.family": FUENTE_UI,
        "font.size": 9,
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
        "axes.prop_cycle": cycler(color=[GRAFICA["metodo"], *ETAPAS]),
        "legend.facecolor": UI["panel_alto"],
        "legend.edgecolor": UI["borde"],
        "legend.labelcolor": UI["texto"],
        "legend.framealpha": 0.9,
    })
    if app:
        mpl.rcParams["toolbar"] = "None"
        for clave in list(mpl.rcParams):
            if clave.startswith("keymap."):
                mpl.rcParams[clave] = []
