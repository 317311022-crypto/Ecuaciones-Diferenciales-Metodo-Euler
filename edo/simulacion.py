"""Una corrida completa, lista para mostrarse paso a paso.

``Simulacion`` es el "modelo" de la app visual, pero no depende de matplotlib:
recibe una :class:`Configuracion` con textos (la ecuación, ``x0``, ``h``...),
la interpreta, integra con el método elegido guardando cada :class:`Paso` con
sus etapas, calcula una solución de referencia para medir el error y deja una
lista de :class:`Aviso` con lo que pudo salir mal (divergencia, inestabilidad,
``f`` no definida...). La interfaz solo se encarga de dibujar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .expresion import ErrorExpresion, Expresion, compilar_edo, compilar_funcion, evaluar_numero
from .metodos import MetodoNumerico, Paso, obtener_metodo
from .solucionador import _construir_malla

LIMITE_DIVERGENCIA = 1e12
"""Si |y| supera este valor consideramos que la solución numérica explotó."""


@dataclass
class Configuracion:
    """Parámetros de una corrida. Los numéricos aceptan expresiones (``"pi/2"``)."""

    ecuacion: str = "x + y"
    x0: float | str = 0.0
    y0: float | str = 1.0
    h: float | str = 0.1
    x_final: float | str = 2.0
    metodo: str = "euler"
    exacta: str = ""
    max_pasos: int = 2000


@dataclass
class Aviso:
    nivel: str          # "info" | "aviso" | "error"
    mensaje: str
    paso: Optional[int] = None


class _Contador:
    """Envuelve f para contar cuántas veces se evalúa."""

    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, t, y):
        self.n += 1
        return self.f(t, y)


@dataclass
class Simulacion:
    """Integra ``y' = f(x, y)`` con la configuración dada (ver módulo)."""

    config: Configuracion
    ecuacion: Expresion = field(init=False)
    exacta: Optional[Expresion] = field(init=False, default=None)
    metodo: MetodoNumerico = field(init=False)
    x0: float = field(init=False)
    y0: float = field(init=False)
    h: float = field(init=False)
    x_final: float = field(init=False)
    pasos: list[Paso] = field(init=False, default_factory=list)
    avisos: list[Aviso] = field(init=False, default_factory=list)
    evaluaciones: int = field(init=False, default=0)
    x_ref: np.ndarray = field(init=False, repr=False)
    y_ref: np.ndarray = field(init=False, repr=False)
    etiqueta_referencia: str = field(init=False, default="")
    pasos_inestables: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        c = self.config
        self.ecuacion = compilar_edo(c.ecuacion)
        self.exacta = compilar_funcion(c.exacta) if c.exacta and c.exacta.strip() else None
        self.metodo = obtener_metodo(c.metodo)
        self.x0 = evaluar_numero(c.x0)
        self.y0 = evaluar_numero(c.y0)
        self.h = evaluar_numero(c.h)
        self.x_final = evaluar_numero(c.x_final)
        if self.h <= 0:
            raise ErrorExpresion("El tamaño de paso h debe ser positivo.", str(c.h))
        if self.x_final <= self.x0:
            raise ErrorExpresion("x final debe ser mayor que x₀.", str(c.x_final))
        n = math.ceil((self.x_final - self.x0) / self.h - 1e-9)
        if n > c.max_pasos:
            raise ErrorExpresion(
                f"Con h = {self.h:g} serían {n} pasos (máximo {c.max_pasos}). Usa un h más grande.", str(c.h))
        self._revisar_unicidad()
        self._integrar()
        self._referencia()

    # ------------------------------------------------------------------ integración
    def _f(self, t, y):
        return self.ecuacion(t, y)

    def _revisar_unicidad(self) -> None:
        """Si ∂f/∂y no está acotada en (x₀, y₀) (f no es Lipschitz), el teorema de
        existencia y unicidad no aplica: puede haber varias soluciones y el método
        solo encontrará una (p. ej. y' = √y, y(0) = 0)."""
        def cociente(d):
            with np.errstate(all="ignore"):
                return abs(float(self._f(self.x0, self.y0 + d)) - float(self._f(self.x0, self.y0))) / d
        try:
            l_grueso, l_fino = cociente(1e-4), cociente(1e-8)
        except (ArithmeticError, ValueError, TypeError):
            return
        if not np.isfinite(l_fino) or (l_fino > 1e3 and l_fino > 50 * max(l_grueso, 1e-12)):
            self.avisos.append(Aviso(
                "info",
                f"∂f/∂y no está acotada cerca de (x₀, y₀) = ({self.x0:g}, {self.y0:g}): f no es Lipschitz ahí, "
                "así que la solución podría no ser única. El método seguirá solo una de ellas.", None))

    def _integrar(self) -> None:
        malla, _ = _construir_malla(self.x0, self.x_final, self.h, None)
        f = _Contador(self._f)
        y = np.asarray(self.y0, dtype=float)
        aviso_inestable = False
        for i in range(len(malla) - 1):
            t, h_i = float(malla[i]), float(malla[i + 1] - malla[i])
            with np.errstate(all="ignore"):
                try:
                    paso = self.metodo.paso_detallado(f, t, y, h_i, indice=i)
                except (ArithmeticError, ValueError, TypeError) as e:
                    self.avisos.append(Aviso("error", f"f no se pudo evaluar cerca de x = {t:g}: {e}", i))
                    break
            pendientes_ok = all(np.all(np.isfinite(e.pendiente)) for e in paso.etapas)
            if not pendientes_ok:
                malas = next(e for e in paso.etapas if not np.all(np.isfinite(e.pendiente)))
                self.avisos.append(Aviso(
                    "error",
                    f"f({malas.t:.4g}, {float(np.ravel(malas.y)[0]):.4g}) = {float(np.ravel(malas.pendiente)[0])}: "
                    "f no está definida o es infinita ahí (¿división entre cero, log o raíz de negativo?).", i))
                break
            if not np.all(np.isfinite(paso.y_siguiente)) or np.any(np.abs(paso.y_siguiente) > LIMITE_DIVERGENCIA):
                self.avisos.append(Aviso(
                    "error", f"y se disparó a {float(np.ravel(paso.y_siguiente)[0]):.3g}: la solución numérica "
                             "diverge (¿asíntota vertical? ¿h demasiado grande?).", i))
                break
            self.pasos.append(paso)

            # ¿El método es estable con este h en esta zona? Ecuación linealizada: λ ≈ ∂f/∂y.
            R = self._factor_amplificacion(t, y, h_i)
            if R is not None:
                lam, factor = R
                if lam < 0 and abs(factor) > 1:
                    self.pasos_inestables.append(i)
                    if not aviso_inestable:
                        aviso_inestable = True
                        self.avisos.append(Aviso(
                            "aviso",
                            f"Inestable: ∂f/∂y ≈ {lam:.3g} y |R(h·∂f/∂y)| = {abs(factor):.3g} > 1. "
                            f"El método amplifica el error; reduce h por debajo de ≈ {self._h_estable(lam):.3g}.", i))
            y = paso.y_siguiente
        self.evaluaciones = f.n

    def _factor_amplificacion(self, t: float, y: np.ndarray, h: float):
        """λ ≈ ∂f/∂y (diferencias centrales) y R(hλ) del método, o ``None``."""
        if np.ndim(y):
            return None
        delta = 1e-6 * max(1.0, abs(float(y)))
        with np.errstate(all="ignore"):
            lam = (float(self._f(t, y + delta)) - float(self._f(t, y - delta))) / (2 * delta)
        if not np.isfinite(lam):
            return None
        R = self.metodo.funcion_estabilidad(h * lam)
        if R is None:
            return None
        return lam, complex(R)

    def _h_estable(self, lam: float) -> float:
        """Mayor h con |R(hλ)| ≤ 1 (búsqueda simple sobre el eje real negativo)."""
        if lam >= 0:
            return math.inf
        z_lim = 0.0
        for z in np.linspace(0, -6, 6001)[1:]:
            if abs(complex(self.metodo.funcion_estabilidad(z))) > 1 + 1e-12:
                break
            z_lim = z
        return float(z_lim / lam)

    # ------------------------------------------------------------------ referencia
    def _referencia(self) -> None:
        """Solución exacta si se dio; si no, RK4 con un paso muchísimo más fino."""
        if self.exacta is not None:
            self.x_ref = np.linspace(self.x0, self.x_final, 800)
            with np.errstate(all="ignore"):
                self.y_ref = np.asarray(self.exacta(self.x_ref), dtype=float) * np.ones_like(self.x_ref)
            self.etiqueta_referencia = "solución exacta"
            return

        rk4 = obtener_metodo("rk4")
        n = max(len(self.pasos), 1)
        sub = max(8, math.ceil(800 / n))
        malla, _ = _construir_malla(self.x0, self.x_final, self.h, None)
        xs, ys = [self.x0], [self.y0]
        y = float(self.y0)
        with np.errstate(all="ignore"):
            for i in range(len(malla) - 1):
                a, b = float(malla[i]), float(malla[i + 1])
                hh = (b - a) / sub
                for j in range(sub):
                    try:
                        y = float(rk4.paso(self._f, a + j * hh, y, hh))
                    except (ArithmeticError, ValueError, TypeError):
                        y = math.nan
                    xs.append(a + (j + 1) * hh)
                    ys.append(y)
                    if not math.isfinite(y) or abs(y) > LIMITE_DIVERGENCIA:
                        break
                if not math.isfinite(y) or abs(y) > LIMITE_DIVERGENCIA:
                    break
        self.x_ref = np.asarray(xs)
        self.y_ref = np.asarray(ys)
        self.etiqueta_referencia = f"referencia RK4 (h/{sub})"
        if not math.isfinite(y) or abs(y) > LIMITE_DIVERGENCIA:
            x_malo = xs[-1]
            ultimo = self.x_nodos[-1]
            if ultimo > x_malo:
                self.avisos.append(Aviso(
                    "aviso",
                    f"La referencia se dispara cerca de x ≈ {x_malo:.3g} (probable asíntota vertical), pero "
                    f"{self.metodo} siguió hasta x = {ultimo:.3g} sin notarlo: esos valores no tienen sentido.", None))

    def referencia_en(self, x) -> np.ndarray:
        """Valor de la referencia en ``x`` (interpolado si es numérica)."""
        x = np.asarray(x, dtype=float)
        if self.exacta is not None:
            with np.errstate(all="ignore"):
                return np.asarray(self.exacta(x), dtype=float) * np.ones_like(x)
        ok = np.isfinite(self.y_ref)
        if ok.sum() < 2:
            return np.full_like(x, np.nan)
        xr, yr = self.x_ref[ok], self.y_ref[ok]
        res = np.interp(x, xr, yr)
        return np.where((x >= xr[0]) & (x <= xr[-1] + 1e-12), res, np.nan)

    # ------------------------------------------------------------------ resultados
    @property
    def n_pasos(self) -> int:
        return len(self.pasos)

    @property
    def x_nodos(self) -> np.ndarray:
        if not self.pasos:
            return np.array([self.x0])
        return np.array([self.pasos[0].t] + [p.t_siguiente for p in self.pasos])

    @property
    def y_nodos(self) -> np.ndarray:
        if not self.pasos:
            return np.array([self.y0])
        return np.array([float(self.pasos[0].y)] + [float(p.y_siguiente) for p in self.pasos])

    @property
    def errores(self) -> np.ndarray:
        """|y_ref(x_n) − y_n| en cada nodo (``nan`` donde no hay referencia)."""
        return np.abs(self.referencia_en(self.x_nodos) - self.y_nodos)

    @property
    def completa(self) -> bool:
        return not any(a.nivel == "error" for a in self.avisos)

    def resumen(self) -> str:
        e = self.errores
        err = f", error final ≈ {e[-1]:.3e}" if np.isfinite(e[-1]) else ""
        return (f"{self.metodo} con h = {self.h:g}: {self.n_pasos} pasos, "
                f"y({self.x_nodos[-1]:g}) ≈ {self.y_nodos[-1]:.6g}{err}")
