"""Funciones de graficación con matplotlib.

Todas aceptan un ``ax`` opcional para poder combinarlas en subgráficas y
devuelven el ``ax`` usado. Los colores salen de ``edo.app.tema`` (el mismo
tema oscuro de la app); llama a ``edo.app.tema.aplicar_tema()`` para usarlo
también en los fondos y textos.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .app.tema import GRAFICA
from .metodos import MetodoNumerico, obtener_metodo
from .problema import ProblemaValorInicial
from .solucionador import Solucion, resolver


def _ax(ax):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    return ax


def _componente(arr: np.ndarray, componente: Optional[int]) -> np.ndarray:
    if arr.ndim == 2:
        return arr[:, 0 if componente is None else componente]
    return arr


def _nombre_componente(problema: ProblemaValorInicial, componente: Optional[int]) -> str:
    if problema.es_sistema:
        i = 0 if componente is None else componente
        return problema.etiquetas[i] if i < len(problema.etiquetas) else f"y[{i}]"
    return problema.etiquetas[0] if problema.etiquetas else "y"


def _malla_exacta(problema: ProblemaValorInicial, puntos: int = 400) -> np.ndarray:
    return np.linspace(problema.t0, problema.t_final, puntos)


def campo_direcciones(
    f,
    t_lim: tuple[float, float],
    y_lim: tuple[float, float],
    ax=None,
    densidad: int = 20,
    color: str = GRAFICA["campo_medio"],
):
    """Dibuja el campo de pendientes de una EDO escalar ``y' = f(t, y)``.

    En cada punto (t, y) se dibuja un segmento corto con pendiente f(t, y):
    las soluciones son curvas tangentes a estos segmentos.
    """
    ax = _ax(ax)
    T, Y = np.meshgrid(np.linspace(*t_lim, densidad), np.linspace(*y_lim, densidad))
    pendiente = np.asarray(f(T, Y), dtype=float) * np.ones_like(T)
    # Normalizamos para que todos los segmentos midan lo mismo en pantalla.
    escala_t = (t_lim[1] - t_lim[0]) / densidad
    escala_y = (y_lim[1] - y_lim[0]) / densidad
    dt = np.ones_like(pendiente) / escala_t
    dy = pendiente / escala_y
    norma = np.hypot(dt, dy)
    ax.quiver(
        T, Y, dt / norma * escala_t, dy / norma * escala_y,
        angles="xy", scale_units="xy", scale=1.6,
        headwidth=0, headlength=0, headaxislength=0,
        color=color, width=0.003, pivot="middle",
    )
    ax.set_xlim(t_lim)
    ax.set_ylim(y_lim)
    return ax


def graficar_solucion(
    sol: Solucion,
    ax=None,
    exacta: bool = True,
    componente: Optional[int] = None,
    marcadores: bool = True,
    **kwargs,
):
    """Grafica la aproximación numérica y, si existe, la solución exacta."""
    ax = _ax(ax)
    p = sol.problema
    nombre = _nombre_componente(p, componente)
    if exacta and p.tiene_solucion_exacta:
        te = _malla_exacta(p)
        ax.plot(te, _componente(p.exacta(te), componente), "-", color=GRAFICA["referencia"], lw=2,
                label=f"{nombre}(t) exacta")
    estilo = {"marker": "o", "ms": 4} if marcadores and sol.n_pasos <= 60 else {}
    estilo.update(kwargs)
    estilo.setdefault("color", GRAFICA["metodo"])
    ax.plot(sol.t, _componente(sol.y, componente), "--", label=f"{sol.nombre_metodo}, h = {sol.h:g}", **estilo)
    ax.set_xlabel("t")
    ax.set_ylabel(nombre)
    ax.set_title(f"{p.nombre}: {p.ecuacion}")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def graficar_pasos_euler(problema: ProblemaValorInicial, h: float, ax=None, n_max: Optional[int] = None):
    """Interpretación geométrica de Euler: sobre el campo de pendientes, cada
    paso avanza en línea recta con la pendiente del punto de partida."""
    if problema.es_sistema:
        raise ValueError("La interpretación geométrica solo aplica a EDOs escalares.")
    ax = _ax(ax)
    sol = resolver(problema, "euler", h=h)
    t, y = sol.t, sol.y
    if n_max is not None:
        t, y = t[: n_max + 1], y[: n_max + 1]

    valores = [y]
    te = _malla_exacta(problema)
    if problema.tiene_solucion_exacta:
        valores.append(problema.exacta(te))
    y_min = min(v.min() for v in valores)
    y_max = max(v.max() for v in valores)
    margen = 0.1 * (y_max - y_min or 1.0)
    campo_direcciones(problema.f, (problema.t0, t[-1]), (y_min - margen, y_max + margen), ax=ax)

    if problema.tiene_solucion_exacta:
        mask = te <= t[-1]
        ax.plot(te[mask], problema.exacta(te[mask]), "-", color=GRAFICA["referencia"], lw=2, label="Solución exacta")
    ax.plot(t, y, "o-", color=GRAFICA["metodo"], lw=2, ms=6, label=f"Euler, h = {h:g}")
    for i in range(len(t) - 1):
        ax.annotate(
            "", xy=(t[i + 1], y[i + 1]), xytext=(t[i], y[i]),
            arrowprops=dict(arrowstyle="->", color=GRAFICA["metodo"], lw=1.5),
        )
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(f"Método de Euler sobre el campo de pendientes\n{problema.ecuacion}")
    ax.legend(loc="best")
    return ax


def comparar_pasos(
    problema: ProblemaValorInicial,
    pasos: Iterable[float],
    metodo: str | MetodoNumerico = "euler",
    ax=None,
    componente: Optional[int] = None,
):
    """Misma EDO y mismo método con distintos tamaños de paso ``h``."""
    ax = _ax(ax)
    met = obtener_metodo(metodo)
    nombre = _nombre_componente(problema, componente)
    if problema.tiene_solucion_exacta:
        te = _malla_exacta(problema)
        ax.plot(te, _componente(problema.exacta(te), componente), "-", color=GRAFICA["referencia"], lw=2.5,
                label="Exacta")
    for h in pasos:
        sol = resolver(problema, met, h=h)
        estilo = {"marker": "o", "ms": 3} if sol.n_pasos <= 40 else {}
        ax.plot(sol.t, _componente(sol.y, componente), "--", label=f"h = {h:g}", **estilo)
    ax.set_xlabel("t")
    ax.set_ylabel(nombre)
    ax.set_title(f"{met}: efecto del tamaño de paso\n{problema.ecuacion}")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def comparar_metodos(
    problema: ProblemaValorInicial,
    metodos: Sequence[str | MetodoNumerico],
    h: float,
    ax=None,
    componente: Optional[int] = None,
):
    """Distintos métodos con el mismo ``h``."""
    ax = _ax(ax)
    nombre = _nombre_componente(problema, componente)
    if problema.tiene_solucion_exacta:
        te = _malla_exacta(problema)
        ax.plot(te, _componente(problema.exacta(te), componente), "-", color=GRAFICA["referencia"], lw=2.5,
                label="Exacta")
    for m in metodos:
        sol = resolver(problema, m, h=h)
        estilo = {"marker": "o", "ms": 3} if sol.n_pasos <= 40 else {}
        ax.plot(sol.t, _componente(sol.y, componente), "--", label=sol.nombre_metodo, **estilo)
    ax.set_xlabel("t")
    ax.set_ylabel(nombre)
    ax.set_title(f"Comparación de métodos (h = {h:g})\n{problema.ecuacion}")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def graficar_error(soluciones: Sequence[Solucion], ax=None, log: bool = True):
    """Error absoluto |y(t_i) - y_i| a lo largo del intervalo."""
    ax = _ax(ax)
    for sol in soluciones:
        ax.plot(sol.t, sol.error_absoluto(), "-o", ms=3, label=f"{sol.nombre_metodo}, h = {sol.h:g}")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("|error|")
    ax.set_title("Error absoluto")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return ax


def graficar_convergencia(
    problema: ProblemaValorInicial,
    metodos: Sequence[str | MetodoNumerico],
    pasos: Sequence[float],
    ax=None,
):
    """Error máximo contra h en escala log-log.

    Un método de orden p aparece como una recta de pendiente p. Se agregan
    rectas de referencia O(h^p) en gris punteado.
    """
    ax = _ax(ax)
    pasos = np.asarray(sorted(pasos, reverse=True), dtype=float)
    for m in metodos:
        met = obtener_metodo(m)
        errores = np.array([resolver(problema, met, h=h).error_maximo() for h in pasos])
        linea, = ax.loglog(pasos, errores, "o-", label=f"{met} (orden {met.orden})")
        ref = errores[0] * (pasos / pasos[0]) ** met.orden
        ax.loglog(pasos, ref, ":", color=linea.get_color(), alpha=0.6)
    ax.set_xlabel("h")
    ax.set_ylabel("error máximo")
    ax.set_title(f"Convergencia (punteado: O(h^p))\n{problema.ecuacion}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    return ax


def graficar_plano_fase(soluciones: Sequence[Solucion], ax=None, exacta: bool = True):
    """Trayectoria en el plano (y[0], y[1]) para sistemas de dos ecuaciones."""
    ax = _ax(ax)
    p = soluciones[0].problema
    if p.dimension != 2:
        raise ValueError("El plano fase requiere un sistema de dos ecuaciones.")
    if exacta and p.tiene_solucion_exacta:
        ye = p.exacta(_malla_exacta(p, 800))
        ax.plot(ye[:, 0], ye[:, 1], "-", color=GRAFICA["referencia"], lw=2, label="Exacta")
    for sol in soluciones:
        ax.plot(sol.y[:, 0], sol.y[:, 1], "-", lw=1.2, label=f"{sol.nombre_metodo}, h = {sol.h:g}")
    ax.plot(*p.y0, "o", color="white", ms=6)
    etiq = p.etiquetas or ("y[0]", "y[1]")
    ax.set_xlabel(etiq[0])
    ax.set_ylabel(etiq[1])
    ax.set_title(f"Plano fase — {p.nombre}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax
