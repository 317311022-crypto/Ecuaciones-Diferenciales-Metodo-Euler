"""``python -m edo.app [ecuacion] [--x0 ..] [--y0 ..] [--h ..] [--hasta ..] [--metodo ..] [--exacta ..] [--calidad ..]``"""

from __future__ import annotations

import argparse

from ..simulacion import Configuracion
from .ventana import main as abrir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m edo.app", description="Método de Euler paso a paso (app visual).")
    p.add_argument("ecuacion", nargs="?", default="x + y", help="lado derecho f(x, y), p. ej. \"x + y\"")
    p.add_argument("--x0", default="0")
    p.add_argument("--y0", default="1")
    p.add_argument("--h", default="0.1")
    p.add_argument("--hasta", default="2", help="x final")
    p.add_argument("--metodo", default="euler")
    p.add_argument("--exacta", default="", help="solución exacta y(x), opcional")
    p.add_argument("--calidad", default="media", choices=["alta", "media", "baja"],
                   help="alta: con parallax · media: laptops (predeterminada) · baja: equipos lentos")
    a = p.parse_args(argv)
    abrir(Configuracion(a.ecuacion, a.x0, a.y0, a.h, a.hasta, a.metodo, a.exacta), calidad=a.calidad)


if __name__ == "__main__":
    main()
