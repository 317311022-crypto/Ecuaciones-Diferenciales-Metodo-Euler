"""Pequeñas utilidades de formato para mostrar números y subíndices."""

from __future__ import annotations

import math

_SUB = str.maketrans("0123456789+-n", "₀₁₂₃₄₅₆₇₈₉₊₋ₙ")


def sub(texto) -> str:
    """Convierte dígitos (y ``n``, ``+``, ``-``) a subíndices: ``sub(12) == '₁₂'``."""
    return str(texto).translate(_SUB)


def etiqueta(texto: str) -> str:
    """``'k1'`` → ``'k₁'``: pasa a subíndice los dígitos finales de una etiqueta."""
    i = len(texto)
    while i > 0 and texto[i - 1].isdigit():
        i -= 1
    return texto[:i] + sub(texto[i:])


def num(valor, decimales: int = 4) -> str:
    """Número con ancho razonable: fijo para valores moderados, científico si no."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "∞" if v > 0 else "-∞"
    if v != 0 and (abs(v) >= 1e5 or abs(v) < 10 ** (-decimales + 1)):
        return f"{v:.{max(decimales - 1, 1)}e}"
    return f"{v:.{decimales}f}"


def nump(valor, decimales: int = 4) -> str:
    """Como :func:`num`, pero entre paréntesis si es negativo: ``0.1·(-2.0000)``."""
    texto = num(valor, decimales)
    return f"({texto})" if texto.startswith("-") else texto


def corto(valor) -> str:
    """Formato compacto (``g``) para parámetros: 0.1, 2, 6.283."""
    v = float(valor)
    return f"{v:.6g}"
