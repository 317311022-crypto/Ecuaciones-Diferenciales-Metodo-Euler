"""Perfiles de calidad visual: cuántos efectos dibujar según el equipo.

Matplotlib dibuja con el **CPU** (con la biblioteca Agg), no con la tarjeta de
video, así que lo que limita la fluidez es cuánto se dibuja en cada cuadro y
cuántos píxeles tiene la ventana. Cada perfil decide:

``decoracion``
    Estrellas, orbes desenfocados y retícula de puntos en el fondo.
``brillo``
    Fracción de las capas de luminiscencia que se dibujan (1 = todas). Con
    menos capas cada una es más intensa, así que el brillo total se conserva.
``resplandor`` / ``pulso``
    Degradado bajo la poligonal y punto actual que "late".
``fps``
    Cuadros por segundo de la animación.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Calidad:
    nombre: str
    descripcion: str
    decoracion: bool
    brillo: float
    resplandor: bool
    pulso: bool
    fps: int

    def halo(self, capas: int, intensidad: float) -> tuple[int, float]:
        """Capas de halo a dibujar y su opacidad, conservando el brillo total."""
        n = max(1, round(capas * self.brillo))
        return n, min(0.35, intensidad * capas / n)

    def halo_puntos(self, capas: int, alcance: float) -> tuple[int, float]:
        """Para puntos: con menos capas se conservan los anillos *interiores* (halo más
        chico pero suave); repartir la misma extensión en pocos anillos los haría notar."""
        n = max(1, round(capas * self.brillo))
        return n, alcance * n / capas

    @property
    def intervalo_ms(self) -> int:
        return max(10, round(1000 / self.fps))


CALIDADES: dict[str, Calidad] = {
    "alta": Calidad(
        "alta", "todas las capas de brillo, 30 fps (computadora de escritorio)",
        decoracion=True, brillo=1.0, resplandor=True, pulso=True, fps=30),
    "media": Calidad(
        "media", "recomendada para laptops: mitad de capas de brillo, 25 fps",
        decoracion=True, brillo=0.5, resplandor=True, pulso=True, fps=25),
    "baja": Calidad(
        "baja", "equipos lentos: fondo liso, brillo mínimo, 20 fps",
        decoracion=False, brillo=0.25, resplandor=False, pulso=False, fps=20),
}


def obtener_calidad(calidad: "str | Calidad") -> Calidad:
    if isinstance(calidad, Calidad):
        return calidad
    clave = str(calidad).strip().lower()
    if clave not in CALIDADES:
        raise ValueError(f"Calidad desconocida '{calidad}'. Opciones: {', '.join(CALIDADES)}")
    return CALIDADES[clave]
