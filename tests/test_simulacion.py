import numpy as np
import pytest

from edo import Configuracion, MetodoNumerico, Simulacion, metodos_disponibles, obtener_metodo, registrar
from edo.expresion import ErrorExpresion


def test_euler_paso_a_paso_coincide_con_burden():
    sim = Simulacion(Configuracion("y - x^2 + 1", 0, 0.5, 0.2, 2, "euler", "(x + 1)^2 - 0.5exp(x)"))
    esperados = [0.5, 0.8, 1.152, 1.5504, 1.98848, 2.458176, 2.9498112, 3.4517734, 3.9501281, 4.4281538, 4.8657845]
    np.testing.assert_allclose(sim.y_nodos, esperados, atol=1e-7)
    p = sim.pasos[1]
    assert len(p.etapas) == 1 and p.etapas[0].etiqueta == "k1"
    assert float(p.etapas[0].pendiente) == pytest.approx(0.8 - 0.04 + 1)
    assert sim.evaluaciones == 10
    assert sim.metodo.formula_numerica(p) == "y₂ = y₁ + h·k₁ = 0.8000 + 0.2000·1.7600 = 1.1520"
    assert sim.completa and not sim.avisos


@pytest.mark.parametrize("nombre", metodos_disponibles())
def test_paso_detallado_igual_a_paso(nombre):
    met = obtener_metodo(nombre)
    f = lambda t, y: np.sin(t) - y**2  # noqa: E731
    detallado = met.paso_detallado(f, 0.3, np.float64(0.7), 0.1)
    assert float(detallado.y_siguiente) == pytest.approx(float(met.paso(f, 0.3, np.float64(0.7), 0.1)))
    etapas_esperadas = {"euler": 1, "heun": 2, "rk4": 4}
    if nombre in etapas_esperadas:
        assert len(detallado.etapas) == etapas_esperadas[nombre]


def test_avisa_inestabilidad():
    sim = Simulacion(Configuracion("-15y", 0, 1, 0.25, 2, "euler", "exp(-15x)"))
    assert sim.pasos_inestables
    assert any(a.nivel == "aviso" and "Inestable" in a.mensaje for a in sim.avisos)
    estable = Simulacion(Configuracion("-15y", 0, 1, 0.1, 2, "euler", "exp(-15x)"))
    assert not estable.pasos_inestables


def test_detecta_explosion_y_asintota():
    sim = Simulacion(Configuracion("y^2", 0, 1, 0.1, 2, "euler"))
    assert not sim.completa
    assert any("asíntota" in a.mensaje for a in sim.avisos)


def test_detecta_f_no_definida():
    sim = Simulacion(Configuracion("1/(x - 1)", 0, 0, 0.25, 2, "euler"))
    assert any(a.nivel == "error" and "no está definida" in a.mensaje for a in sim.avisos)


def test_avisa_falta_de_unicidad():
    sim = Simulacion(Configuracion("sqrt(y)", 0, 0, 0.1, 1, "euler"))
    assert any(a.nivel == "info" and "Lipschitz" in a.mensaje for a in sim.avisos)
    assert np.all(sim.y_nodos == 0)


def test_referencia_rk4_cuando_no_hay_exacta():
    sim = Simulacion(Configuracion("x + y", 0, 1, 0.1, 1, "euler"))
    exacta = 2 * np.exp(sim.x_nodos) - sim.x_nodos - 1
    np.testing.assert_allclose(sim.referencia_en(sim.x_nodos), exacta, rtol=1e-8)
    assert sim.errores[-1] == pytest.approx(abs(exacta[-1] - sim.y_nodos[-1]), rel=1e-6)


@pytest.mark.parametrize("cfg", [
    Configuracion("x + y", 0, 1, -0.1, 1),
    Configuracion("x + y", 1, 1, 0.1, 0),
    Configuracion("x + y", 0, 1, 1e-6, 1),
    Configuracion("x + q", 0, 1, 0.1, 1),
])
def test_configuraciones_invalidas(cfg):
    with pytest.raises(ErrorExpresion):
        Simulacion(cfg)


def test_metodo_nuevo_aparece_con_sus_etapas():
    @registrar
    class PuntoMedioTraza(MetodoNumerico):
        nombre = "punto_medio_traza"
        orden = 2

        def paso(self, f, t, y, h):
            k1 = self.evaluar(f, t, y, "k1")
            k2 = self.evaluar(f, t + h / 2, y + h / 2 * k1, "k2")
            return y + h * k2

    sim = Simulacion(Configuracion("x + y", 0, 1, 0.1, 1, "punto_medio_traza"))
    assert [e.etiqueta for e in sim.pasos[0].etapas] == ["k1", "k2"]
    assert sim.pasos[0].etapas[1].t == pytest.approx(0.05)
