import numpy as np
import pytest

from edo import MetodoNumerico, ejemplos, metodos_disponibles, obtener_metodo, orden_observado, registrar, resolver


def test_metodos_incluidos_registrados():
    assert {"euler", "heun", "rk4"} <= set(metodos_disponibles())


def test_metodo_desconocido():
    with pytest.raises(KeyError):
        obtener_metodo("no-existe")


@pytest.mark.parametrize("nombre", ["euler", "heun", "rk4"])
def test_orden_observado_coincide_con_teorico(nombre):
    met = obtener_metodo(nombre)
    filas = orden_observado(ejemplos.ejemplo_burden(), nombre, [0.02, 0.01, 0.005])
    assert filas[-1][2] == pytest.approx(met.orden, abs=0.1)


def test_metodos_de_mayor_orden_son_mas_precisos():
    pvi = ejemplos.crecimiento_exponencial()
    errores = [resolver(pvi, m, h=0.1).error_maximo() for m in ("euler", "heun", "rk4")]
    assert errores[0] > errores[1] > errores[2]


def test_registrar_metodo_nuevo():
    @registrar
    class PuntoMedio(MetodoNumerico):
        nombre = "punto_medio_prueba"
        orden = 2

        def paso(self, f, t, y, h):
            return y + h * np.asarray(f(t + h / 2, y + h / 2 * np.asarray(f(t, y))))

    assert "punto_medio_prueba" in metodos_disponibles()
    sol = resolver(ejemplos.ejemplo_burden(), "punto_medio_prueba", h=0.1)
    assert sol.error_maximo() < resolver(ejemplos.ejemplo_burden(), "euler", h=0.1).error_maximo()


def test_registrar_rechaza_clases_invalidas():
    with pytest.raises(TypeError):
        registrar(object)
