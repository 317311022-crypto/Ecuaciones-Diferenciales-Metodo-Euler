import numpy as np
import pytest

from edo import ProblemaValorInicial, ejemplos, resolver
from edo.metodos import Euler


def test_un_paso_a_mano():
    # y' = y, y(0) = 1, h = 0.1  ->  y1 = 1 + 0.1·1 = 1.1
    pvi = ProblemaValorInicial(f=lambda t, y: y, t0=0, y0=1, t_final=0.1)
    sol = resolver(pvi, "euler", h=0.1)
    assert sol.y[1] == pytest.approx(1.1)


def test_valores_de_referencia_burden():
    # Tabla 5.1 de Burden & Faires (y' = y - t^2 + 1, y(0) = 0.5, h = 0.2)
    esperados = [0.5, 0.8, 1.152, 1.5504, 1.98848, 2.458176,
                 2.9498112, 3.4517734, 3.9501281, 4.4281538, 4.8657845]
    sol = resolver(ejemplos.ejemplo_burden(), "euler", h=0.2)
    np.testing.assert_allclose(sol.y, esperados, atol=1e-7)
    np.testing.assert_allclose(sol.t, np.linspace(0, 2, 11))


def test_euler_es_orden_uno():
    pvi = ejemplos.ejemplo_burden()
    e1 = resolver(pvi, "euler", h=0.01).error_maximo()
    e2 = resolver(pvi, "euler", h=0.005).error_maximo()
    assert e1 / e2 == pytest.approx(2.0, rel=0.05)


def test_h_o_n_equivalentes():
    pvi = ejemplos.ejemplo_burden()
    a = resolver(pvi, "euler", h=0.25)
    b = resolver(pvi, "euler", n=8)
    np.testing.assert_allclose(a.y, b.y)


def test_h_que_no_divide_el_intervalo_termina_en_t_final():
    sol = resolver(ejemplos.ejemplo_burden(), "euler", h=0.3)
    assert sol.t[-1] == pytest.approx(2.0)
    assert np.all(np.diff(sol.t) > 0)


def test_sistema_oscilador():
    pvi = ejemplos.oscilador_armonico()
    sol = resolver(pvi, Euler(), h=0.01)
    assert sol.y.shape == (len(sol.t), 2)
    # Euler explícito gana energía en el oscilador: la amplitud crece.
    energia = sol.y[:, 0] ** 2 + sol.y[:, 1] ** 2
    assert energia[-1] > energia[0]


def test_parametros_invalidos():
    pvi = ejemplos.ejemplo_burden()
    with pytest.raises(ValueError):
        resolver(pvi, "euler")  # falta h o n
    with pytest.raises(ValueError):
        resolver(pvi, "euler", h=0.1, n=10)  # ambos
    with pytest.raises(ValueError):
        resolver(pvi, "euler", h=-0.1)
