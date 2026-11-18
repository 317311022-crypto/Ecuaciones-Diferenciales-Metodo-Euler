"""Pruebas de la app sin ventana (backend Agg), simulando eventos de ratón y teclado."""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402

from edo import Configuracion  # noqa: E402
from edo.app import EulerLab  # noqa: E402
from edo.app.consola import colorear  # noqa: E402


@pytest.fixture
def app():
    a = EulerLab(Configuracion("x + y", 0, 1, 0.1, 1, "euler", "2exp(x) - x - 1"), animar=False)
    yield a
    import matplotlib.pyplot as plt

    plt.close(a.fig)


def _clic(app, ax, fx, fy, boton=1, doble=False):
    x, y = ax.transAxes.transform((fx, fy))
    MouseEvent("button_press_event", app.fig.canvas, x, y, boton, dblclick=doble)._process()
    MouseEvent("button_release_event", app.fig.canvas, x, y, boton)._process()


def _tecla(app, tecla):
    KeyEvent("key_press_event", app.fig.canvas, tecla)._process()


def test_estado_inicial_muestra_todo(app):
    assert app.sim.n_pasos == 10
    assert app.revelados == 10 and app.seleccion == 9
    texto = "\n".join(l.texto for l in app.consola.lineas)
    assert "paso 0" in texto and "listo" in texto and "y₁ = y₀ + h·k₁" in texto


def test_animacion_completa(app):
    app.alternar_animacion()
    assert app.animando and app.revelados == 1
    for _ in range(500):
        app._t_prev -= 0.05
        app._tick()
        if not app.animando:
            break
    assert not app.animando and app.revelados == 10 and app.progreso == 1.0


def test_navegacion_con_teclado(app):
    _tecla(app, "home")
    assert (app.revelados, app.seleccion) == (1, 0)
    _tecla(app, "right")
    _tecla(app, "right")
    assert app.seleccion == 2
    _tecla(app, "left")
    assert app.seleccion == 1
    _tecla(app, "end")
    assert app.revelados == 10
    _tecla(app, "m")                    # siguiente método
    assert app.sim.metodo.nombre != "euler"


def test_escribir_ecuacion_y_resolver(app):
    caja = app.cajas["ecuacion"]
    _clic(app, caja.ax, 0.5, 0.5)
    assert caja.capturekeystrokes
    caja.set_val("")
    for c in "y(1-y)":
        _tecla(app, c)
    _tecla(app, "enter")
    assert app.sim is not None and app.sim.ecuacion.original == "y(1-y)"


def test_error_de_entrada_no_rompe(app):
    app.cajas["ecuacion"].set_val("x + sin y")
    app.resolver()
    assert app.sim is None
    assert any("paréntesis" in l.texto for l in app.consola.lineas)
    _tecla(app, "right")                # no debe fallar sin simulación
    app.cajas["ecuacion"].set_val("x + sin(y)")
    app.resolver(animar=False)
    assert app.sim is not None


def test_clic_en_registro_selecciona_paso(app):
    app.ir_a(10, 0)
    i = next(i for i, l in enumerate(app.consola.lineas) if l.paso == 0)
    app.consola.inicio = i
    app.consola.renderizar()
    y = 1 - app.consola.margen_sup - 0.5 * app.consola.fila
    app.seleccion = 5
    _clic(app, app.consola.ax, 0.5, y)
    assert app.seleccion == 0


def test_zoom_y_arrastre(app):
    ax = app.grafica.ax
    antes = ax.get_xlim()
    x, y = ax.transAxes.transform((0.5, 0.5))
    MouseEvent("scroll_event", app.fig.canvas, x, y, step=2)._process()
    assert (ax.get_xlim()[1] - ax.get_xlim()[0]) < (antes[1] - antes[0])
    MouseEvent("button_press_event", app.fig.canvas, x, y, 1)._process()
    MouseEvent("motion_notify_event", app.fig.canvas, x + 80, y + 30)._process()
    MouseEvent("button_release_event", app.fig.canvas, x + 80, y + 30, 1)._process()
    assert ax.get_xlim()[0] != pytest.approx(antes[0]) and ax.get_ylim() != pytest.approx(app.grafica.vista_inicial[1])
    for _ in range(80):
        app._t_prev -= 0.05
        app._tick()
    _clic(app, ax, 0.5, 0.5, doble=True)
    for _ in range(200):
        app._t_prev -= 0.05
        app._tick()
    assert ax.get_xlim() == pytest.approx(app.grafica.vista_inicial[0], rel=1e-3)


def test_modo_seguir(app):
    app.alternar_seguir()
    app.ir_a(5, 4)
    for _ in range(100):
        app._t_prev -= 0.05
        app._tick()
    x0, x1 = app.grafica.ax.get_xlim()
    assert x0 < app.sim.x_nodos[5] < x1
    assert (x1 - x0) < 0.6 * (app.grafica.vista_inicial[0][1] - app.grafica.vista_inicial[0][0])


def test_guardar_captura(app, tmp_path):
    ruta = app.ir_a(4, 3, 0.5).guardar(str(tmp_path / "captura.png"), dpi=60)
    assert (tmp_path / "captura.png").stat().st_size > 10_000
    assert ruta.endswith(".png")


def test_colorear_tokens():
    tokens = colorear("y₁ = y₀ + h·k₁ = 1.0000 + 0.1000·1.0000")
    estilos = {t: e for t, e in tokens}
    assert estilos["k₁"] == "k1"
    assert estilos["1.0000"] == "numero"
    assert "".join(t for t, _ in tokens) == "y₁ = y₀ + h·k₁ = 1.0000 + 0.1000·1.0000"
