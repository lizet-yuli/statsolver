"""StatSolver: reconocimiento y solución paso a paso de ejercicios estadísticos."""
from .models import Ejercicio, Solucion
from .parser import interpretar_texto
from .solvers import resolver

__all__ = ["Ejercicio", "Solucion", "interpretar_texto", "resolver"]
