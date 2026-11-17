"""Registro de métodos numéricos.

Permite elegir un método por su nombre (``"euler"``, ``"rk4"``, ...) sin
que el resto del código tenga que conocer las clases concretas. Así,
cambiar de método es cambiar una cadena de texto.
"""

from __future__ import annotations

from typing import Type, TypeVar

from .base import MetodoNumerico

_REGISTRO: dict[str, Type[MetodoNumerico]] = {}

M = TypeVar("M", bound=Type[MetodoNumerico])


def registrar(cls: M) -> M:
    """Decorador que agrega un método al registro usando ``cls.nombre``."""
    if not (isinstance(cls, type) and issubclass(cls, MetodoNumerico)):
        raise TypeError("Solo se pueden registrar subclases de MetodoNumerico.")
    if not cls.nombre:
        raise ValueError(f"{cls.__name__} debe definir el atributo 'nombre'.")
    _REGISTRO[cls.nombre.lower()] = cls
    return cls


def obtener_metodo(metodo: str | MetodoNumerico | Type[MetodoNumerico]) -> MetodoNumerico:
    """Devuelve una instancia del método pedido.

    Acepta el nombre registrado (``"euler"``), una clase o una instancia.
    """
    if isinstance(metodo, MetodoNumerico):
        return metodo
    if isinstance(metodo, type) and issubclass(metodo, MetodoNumerico):
        return metodo()
    if isinstance(metodo, str):
        clave = metodo.lower()
        if clave not in _REGISTRO:
            disponibles = ", ".join(metodos_disponibles())
            raise KeyError(f"Método '{metodo}' no registrado. Disponibles: {disponibles}")
        return _REGISTRO[clave]()
    raise TypeError(f"No se reconoce el método: {metodo!r}")


def metodos_disponibles() -> list[str]:
    """Nombres de todos los métodos registrados."""
    return sorted(_REGISTRO)
