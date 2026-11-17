import numpy as np
import pytest

from edo.expresion import ErrorExpresion, compilar_edo, compilar_funcion, evaluar_numero


@pytest.mark.parametrize("texto, esperado", [
    ("x + y", 3.0),
    ("y' = x + y", 3.0),
    ("dy/dx = x*y", 2.0),
    ("2x + 3y", 8.0),            # multiplicación implícita
    ("xy", 2.0),                 # x*y
    ("y^2 - x", 3.0),            # ^ es potencia, no XOR
    ("y**2 - x", 3.0),
    ("3(x + 1)", 6.0),
    ("sen(0) + cos(0)", 1.0),    # 'sen' en español
    ("e^0 + pi - π", 1.0),
    ("y·x", 2.0),
    ("t + y", 3.0),              # t es sinónimo de x
])
def test_expresiones_validas(texto, esperado):
    assert compilar_edo(texto)(1.0, 2.0) == pytest.approx(esperado)


@pytest.mark.parametrize("texto", [
    "", "x + ", "sin x", "z + y", "__import__('os')", "x.real", "lambda: 1", "open('a')", "(x + 1",
    "[1, 2]", "x if y else 1",
])
def test_expresiones_invalidas(texto):
    with pytest.raises(ErrorExpresion):
        compilar_edo(texto)


def test_es_vectorizada_y_acepta_constantes():
    x = np.linspace(0, 1, 5)
    assert compilar_edo("x*y")(x, np.ones(5)).shape == (5,)
    assert np.all(np.broadcast_to(compilar_edo("2")(x, x), (5,)) == 2)


def test_semantica_numerica_de_numpy():
    # Nada de excepciones de Python: inf / nan como en NumPy.
    assert np.isinf(compilar_edo("1/x")(0.0, 0.0))
    assert np.isnan(compilar_edo("sqrt(y)")(0.0, -1.0))
    assert np.isinf(compilar_edo("9^9^9")(0.0, 0.0))    # no se congela calculando un entero gigante


def test_posicion_del_error():
    with pytest.raises(ErrorExpresion) as info:
        compilar_edo("x + sin y")
    assert info.value.posicion == 4


def test_funcion_exacta_y_parametros():
    assert compilar_funcion("y(x) = 2exp(x) - x - 1")(0.0) == pytest.approx(1.0)
    assert evaluar_numero("pi/2") == pytest.approx(np.pi / 2)
    assert evaluar_numero(0.1) == 0.1
    with pytest.raises(ErrorExpresion):
        evaluar_numero("x + 1")
