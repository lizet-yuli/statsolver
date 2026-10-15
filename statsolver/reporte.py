"""Exporta una solución como documento Markdown (con LaTeX)."""
from __future__ import annotations

from .models import TIPOS, Ejercicio, Solucion
from .solvers import ft


def a_markdown(e: Ejercicio, sol: Solucion) -> str:
    lineas = [f"# {sol.titulo}", "", f"**Tipo:** {TIPOS.get(e.tipo, e.tipo)}", ""]
    if e.transcripcion:
        lineas += ["## Enunciado", "", e.transcripcion, ""]
    if e.latex:
        lineas += [f"$${e.latex}$$", ""]
    lineas += ["## Solución paso a paso", ""]
    for i, paso in enumerate(sol.pasos, 1):
        lineas += [f"### {i}. {paso.titulo}", ""]
        if paso.texto:
            lineas += [paso.texto, ""]
        for ltx in paso.latex:
            lineas += [f"$${ltx}$$", ""]
        if paso.tabla is not None:
            lineas += [paso.tabla.to_markdown(index=False) if hasattr(paso.tabla, "to_markdown") else str(paso.tabla), ""]
    if sol.resultados:
        lineas += ["## Resultados", "", "| Medida | Valor |", "|---|---|"]
        for k, v in sol.resultados.items():
            lineas.append(f"| {k} | {ft(v, 6) if isinstance(v, (int, float)) else v} |")
        lineas.append("")
    if sol.conclusion:
        lineas += ["## Conclusión", "", sol.conclusion.replace("**", ""), ""]
    for adv in sol.advertencias:
        lineas += [f"> ⚠️ {adv}", ""]
    lineas += ["---", "*Generado con StatSolver.*"]
    return "\n".join(lineas)
