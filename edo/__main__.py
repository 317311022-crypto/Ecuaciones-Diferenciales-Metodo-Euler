"""Línea de comandos.

Ejemplos::

    python -m edo --listar
    python -m edo --problema burden --metodo euler --h 0.2
    python -m edo --problema enfriamiento --metodo rk4 --n 30 --grafica salida.png
    python -m edo --problema burden --convergencia
"""

from __future__ import annotations

import argparse
import sys

from . import ejemplos
from .metodos import metodos_disponibles
from .solucionador import orden_observado, resolver


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m edo",
        description="Resuelve problemas de valor inicial con métodos numéricos (Euler por defecto).",
    )
    parser.add_argument("--problema", choices=sorted(ejemplos.PROBLEMAS), default="burden")
    parser.add_argument("--metodo", default="euler", help="Método numérico (ver --listar).")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--h", type=float, help="Tamaño de paso.")
    grupo.add_argument("--n", type=int, help="Número de pasos.")
    parser.add_argument("--cada", type=int, default=1, help="Mostrar solo cada k-ésimo renglón de la tabla.")
    parser.add_argument("--grafica", metavar="ARCHIVO", help="Guardar la gráfica en un archivo (png, pdf, svg).")
    parser.add_argument("--mostrar", action="store_true", help="Abrir la gráfica en una ventana.")
    parser.add_argument("--convergencia", action="store_true", help="Estimar el orden de convergencia del método.")
    parser.add_argument("--listar", action="store_true", help="Listar problemas y métodos disponibles.")
    args = parser.parse_args(argv)

    if args.listar:
        print("Métodos:  ", ", ".join(metodos_disponibles()))
        print("Problemas:", ", ".join(sorted(ejemplos.PROBLEMAS)))
        return 0

    problema = ejemplos.PROBLEMAS[args.problema]()
    print(problema, end="\n\n")

    if args.convergencia:
        print(f"Orden de convergencia observado — método: {args.metodo}")
        print(f"{'h':>10} {'error máx':>14} {'orden p':>10}")
        for h, err, p in orden_observado(problema, args.metodo, [0.2, 0.1, 0.05, 0.025, 0.0125]):
            orden = "—" if p != p else f"{p:.4f}"
            print(f"{h:>10g} {err:>14.6e} {orden:>10}")
        return 0

    if args.h is None and args.n is None:
        args.h = 0.1
    try:
        sol = resolver(problema, args.metodo, h=args.h, n=args.n)
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Método: {sol.nombre_metodo}   ({sol.metodo.formula})")
    print(f"h = {sol.h:g},  pasos = {sol.n_pasos}\n")
    print(sol.tabla(cada=args.cada))
    if problema.tiene_solucion_exacta:
        print(f"\nError máximo: {sol.error_maximo():.6e}")

    if args.grafica or args.mostrar:
        import matplotlib

        if not args.mostrar:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .app.tema import aplicar_tema
        from .graficas import graficar_plano_fase, graficar_solucion

        aplicar_tema()

        if problema.es_sistema and problema.dimension == 2:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            graficar_solucion(sol, ax=ax1)
            graficar_plano_fase([sol], ax=ax2)
        else:
            graficar_solucion(sol)
        plt.tight_layout()
        if args.grafica:
            plt.savefig(args.grafica, dpi=150)
            print(f"\nGráfica guardada en {args.grafica}")
        if args.mostrar:
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
