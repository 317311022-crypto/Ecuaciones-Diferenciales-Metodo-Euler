"""Convierte texto como ``"x + y^2"`` en una función de Python, de forma segura.

La app visual deja escribir la ecuación diferencial como texto. Usar
``eval`` directamente sería peligroso (``eval`` ejecuta *cualquier* código) y
poco amable (``^`` en Python es XOR, no potencia). Aquí:

1. Se normaliza la notación matemática: ``y' =`` / ``dy/dx =`` al inicio,
   ``^`` → ``**``, ``²`` → ``**2``, ``·`` → ``*``, ``sen`` → ``sin``, ``π`` → ``pi``.
2. Se agrega la multiplicación implícita: ``2x`` → ``2*x``, ``3(x+1)`` → ``3*(x+1)``,
   ``xy`` → ``x*y``.
3. Se analiza con el módulo ``ast`` y **solo** se aceptan números, variables,
   operaciones aritméticas y funciones de una lista blanca (``sin``, ``exp``...).
4. Los enteros se convierten a flotantes para que ``9^9^9`` dé ``inf`` en vez
   de congelar el programa calculando un entero gigante.

El resultado usa NumPy, así que funciona con escalares y con arreglos (lo que
permite dibujar el campo de pendientes de un solo golpe).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class ErrorExpresion(ValueError):
    """La expresión no se pudo interpretar. ``posicion`` apunta al problema (o ``None``)."""

    def __init__(self, mensaje: str, texto: str = "", posicion: int | None = None):
        super().__init__(mensaje)
        self.texto = texto
        self.posicion = posicion


def _sec(u):
    return 1 / np.cos(u)


def _csc(u):
    return 1 / np.sin(u)


def _cot(u):
    return 1 / np.tan(u)


FUNCIONES: dict[str, Callable] = {
    "sin": np.sin, "sen": np.sin, "cos": np.cos, "tan": np.tan, "tg": np.tan,
    "sec": _sec, "csc": _csc, "cot": _cot,
    "asin": np.arcsin, "arcsin": np.arcsin, "acos": np.arccos, "arccos": np.arccos,
    "atan": np.arctan, "arctan": np.arctan,
    "sinh": np.sinh, "senh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "raiz": np.sqrt, "cbrt": np.cbrt, "abs": np.abs, "sign": np.sign,
    "min": np.minimum, "max": np.maximum,
}
"""Funciones permitidas en las expresiones."""

CONSTANTES: dict[str, float] = {"pi": float(np.pi), "e": float(np.e)}

_PREFIJO = re.compile(
    r"""^\s*(?:
        [a-zA-Z]\s*(?:'|′|’)\s*(?:\(\s*[a-zA-Z]\s*\))? |    # y'   y'(x)
        d\s*[a-zA-Z]\s*/\s*d\s*[a-zA-Z] |                   # dy/dx
        f\s*\(\s*[a-zA-Z]\s*,\s*[a-zA-Z]\s*\) |             # f(x, y)
        [a-zA-Z]\s*\(\s*[a-zA-Z]\s*\)                       # y(x)   (solución exacta)
    )\s*=\s*""",
    re.VERBOSE,
)

_REEMPLAZOS = [
    ("·", "*"), ("×", "*"), ("÷", "/"), ("−", "-"), ("–", "-"),
    ("²", "^2"), ("³", "^3"), ("π", "pi"), ("√", "sqrt"), ("**", "^"),
]

_TOKEN = re.compile(
    r"""\s*(?:
        (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?) |
        (?P<nom>[A-Za-z_][A-Za-z_0-9]*) |
        (?P<op>[-+*/%^(),])
    )""",
    re.VERBOSE,
)

_NODOS_PERMITIDOS = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
)


@dataclass
class Expresion:
    """Expresión ya validada y compilada. Se llama como ``expr(x, y)``."""

    original: str
    python: str
    variables: tuple[str, ...]
    usadas: frozenset[str] = field(default_factory=frozenset)
    _codigo: object = field(default=None, repr=False)
    _globales: dict = field(default_factory=dict, repr=False)

    def __call__(self, *valores):
        if len(valores) != len(self.variables):
            raise TypeError(f"Se esperaban {len(self.variables)} valores: {', '.join(self.variables)}")
        # Escalares como np.float64: así 1/0 da inf y (-1)**0.5 da nan (semántica de
        # NumPy) en vez de lanzar una excepción o devolver un complejo.
        entorno = {
            nombre: np.float64(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float)
            for nombre, v in zip(self.variables, valores)
        }
        # 'x' y 't' son la misma variable independiente.
        if "x" in entorno and "t" not in entorno:
            entorno["t"] = entorno["x"]
        elif "t" in entorno and "x" not in entorno:
            entorno["x"] = entorno["t"]
        # Sin builtins: el código compilado no puede llegar a nada fuera de la lista blanca.
        with np.errstate(all="ignore"):
            return eval(self._codigo, self._globales, entorno)

    def __str__(self) -> str:
        return self.original


def _normalizar(texto: str) -> str:
    texto = texto.strip()
    texto = _PREFIJO.sub("", texto, count=1)
    for viejo, nuevo in _REEMPLAZOS:
        texto = texto.replace(viejo, nuevo)
    return texto.strip()


def _tokenizar(texto: str) -> list[tuple[str, str, int]]:
    tokens, pos = [], 0
    while pos < len(texto):
        if texto[pos:].strip() == "":
            break
        m = _TOKEN.match(texto, pos)
        if not m or m.end() == pos:
            j = pos
            while j < len(texto) and texto[j].isspace():
                j += 1
            raise ErrorExpresion(f"Carácter no válido «{texto[j]}».", texto, j)
        tipo = m.lastgroup
        tokens.append((tipo, m.group(tipo), m.start(tipo)))
        pos = m.end()
    return tokens


def _multiplicacion_implicita(tokens, variables: tuple[str, ...], texto: str) -> str:
    """Une los tokens agregando ``*`` donde la notación matemática lo omite."""
    simples = set(variables) | {"x", "y", "t"}
    expandidos: list[tuple[str, str, int]] = []
    for tipo, valor, pos in tokens:
        # "xy" → x*y, "2xy" ya llega como 2, xy. Solo con variables de una letra.
        if tipo == "nom" and valor not in FUNCIONES and valor not in CONSTANTES and valor not in variables \
                and len(valor) > 1 and set(valor) <= simples:
            for i, letra in enumerate(valor):
                expandidos.append(("nom", letra, pos + i))
        else:
            expandidos.append((tipo, valor, pos))

    partes: list[str] = []
    anterior = None
    for i, (tipo, valor, pos) in enumerate(expandidos):
        if tipo == "nom" and valor in FUNCIONES:
            siguiente = expandidos[i + 1] if i + 1 < len(expandidos) else None
            if siguiente is None or siguiente[1] != "(":
                raise ErrorExpresion(f"La función '{valor}' necesita paréntesis: {valor}(...)", texto, pos)
        if anterior is not None:
            a_tipo, a_valor = anterior
            termina_valor = a_tipo == "num" or a_valor == ")" or (a_tipo == "nom" and a_valor not in FUNCIONES)
            empieza_valor = tipo in ("num", "nom") or valor == "("
            if termina_valor and empieza_valor:
                partes.append("*")
        partes.append("**" if valor == "^" else valor)
        anterior = (tipo, valor)
    return " ".join(partes)


class _Validador(ast.NodeTransformer):
    def __init__(self, permitidas: set[str], texto: str):
        self.permitidas = permitidas
        self.texto = texto
        self.usadas: set[str] = set()
        self.constantes: dict[str, np.float64] = {}

    def generic_visit(self, node):
        if not isinstance(node, _NODOS_PERMITIDOS):
            raise ErrorExpresion(f"No se permite '{type(node).__name__}' en la expresión.", self.texto)
        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCIONES:
            raise ErrorExpresion("Solo se pueden llamar funciones matemáticas conocidas.", self.texto)
        if node.keywords:
            raise ErrorExpresion("Las funciones no aceptan argumentos con nombre.", self.texto)
        node.args = [self.visit(a) for a in node.args]
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in FUNCIONES:
            raise ErrorExpresion(f"'{node.id}' es una función: escribe {node.id}(...)", self.texto)
        if node.id not in self.permitidas and node.id not in CONSTANTES:
            sugerencia = ""
            if set(node.id) <= set("xyt") or len(node.id) > 1:
                sugerencia = " ¿Falta un '*' entre variables?"
            validas = ", ".join(sorted(self.permitidas))
            raise ErrorExpresion(f"Nombre desconocido '{node.id}'. Variables válidas: {validas}.{sugerencia}", self.texto)
        self.usadas.add(node.id)
        return node

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ErrorExpresion("Solo se permiten constantes numéricas.", self.texto)
        # Cada constante se vuelve un np.float64 con nombre interno (_c0, _c1...):
        # así toda la aritmética es de NumPy y 9^9^9 da inf en lugar de congelarse
        # (con enteros) o lanzar OverflowError (con floats de Python).
        nombre = f"_c{len(self.constantes)}"
        self.constantes[nombre] = np.float64(node.value)
        return ast.copy_location(ast.Name(id=nombre, ctx=ast.Load()), node)


def compilar(texto: str, variables: tuple[str, ...] = ("x", "y")) -> Expresion:
    """Valida y compila ``texto`` como función de ``variables``.

    ``x`` y ``t`` se aceptan indistintamente como variable independiente.

    >>> f = compilar("y' = x + y^2")
    >>> f(1.0, 2.0)
    5.0
    """
    if not isinstance(texto, str) or not texto.strip():
        raise ErrorExpresion("La expresión está vacía.", texto or "")
    normal = _normalizar(texto)
    if not normal:
        raise ErrorExpresion("Falta el lado derecho de la ecuación.", texto)
    tokens = _tokenizar(normal)
    python = _multiplicacion_implicita(tokens, variables, normal)

    permitidas = set(variables)
    if "x" in permitidas or "t" in permitidas:
        permitidas |= {"x", "t"}
    try:
        arbol = ast.parse(python, mode="eval")
    except SyntaxError as e:
        raise ErrorExpresion("Sintaxis incompleta o inválida (¿paréntesis sin cerrar? ¿operador al final?).",
                             normal, None) from e
    validador = _Validador(permitidas, normal)
    arbol = ast.fix_missing_locations(validador.visit(arbol))
    codigo = compile(arbol, "<expresion>", "eval")
    globales = {"__builtins__": {}, **FUNCIONES, **{k: np.float64(v) for k, v in CONSTANTES.items()},
                **validador.constantes}
    expr = Expresion(original=normal, python=python, variables=tuple(variables),
                     usadas=frozenset(validador.usadas), _codigo=codigo, _globales=globales)
    # Prueba rápida: detecta errores como sin(1, 2) antes de usarla.
    try:
        with np.errstate(all="ignore"):
            expr(*([0.5] * len(variables)))
    except TypeError as e:
        raise ErrorExpresion(f"Argumentos inválidos: {e}", normal) from e
    except ArithmeticError:
        pass  # p. ej. 1/(x - 0.5): es válida, solo no está definida en ese punto
    return expr


def compilar_edo(texto: str) -> Expresion:
    """Lado derecho ``f(x, y)`` de ``y' = f(x, y)``."""
    return compilar(texto, ("x", "y"))


def compilar_funcion(texto: str) -> Expresion:
    """Función de una variable, p. ej. la solución exacta ``y(x)``."""
    return compilar(texto, ("x",))


def evaluar_numero(texto: str | float) -> float:
    """Evalúa un parámetro numérico que puede ser una expresión: ``"pi/2"``, ``"1e-3"``."""
    if isinstance(texto, (int, float, np.floating, np.integer)):
        return float(texto)
    expr = compilar(str(texto), ())
    try:
        with np.errstate(all="ignore"):
            valor = float(expr())
    except ArithmeticError:
        valor = float("nan")
    if not np.isfinite(valor):
        raise ErrorExpresion(f"'{texto}' no es un número finito.", str(texto))
    return valor
