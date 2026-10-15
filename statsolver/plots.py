"""Gráficos de apoyo para cada tipo de ejercicio."""
from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

AZUL = "#2563eb"
NARANJA = "#ea580c"
GRIS = "#94a3b8"
VERDE = "#16a34a"


def _fig(titulo: str, ancho: float = 7, alto: float = 3.6):
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=110)
    ax.set_title(titulo, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def histograma(datos: Sequence[float], media: float, mediana: float):
    fig, ax = _fig("Histograma de los datos")
    bins = "sturges" if len(set(datos)) > 8 else np.arange(min(datos) - 0.5, max(datos) + 1.5, 1)
    ax.hist(datos, bins=bins, color=AZUL, alpha=0.8, edgecolor="white")
    ax.axvline(media, color=NARANJA, lw=2, label=f"Media = {media:.4g}")
    ax.axvline(mediana, color=VERDE, lw=2, ls="--", label=f"Mediana = {mediana:.4g}")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Frecuencia")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def caja(datos: Sequence[float]):
    fig, ax = _fig("Diagrama de caja", alto=2.4)
    ax.boxplot(datos, orientation="horizontal", widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#dbeafe", color=AZUL),
               medianprops=dict(color=NARANJA, lw=2),
               whiskerprops=dict(color=AZUL), capprops=dict(color=AZUL),
               flierprops=dict(marker="o", markerfacecolor=NARANJA, markeredgecolor=NARANJA))
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("Valor")
    fig.tight_layout()
    return fig


def discreta(ks: Sequence[int], probs: Sequence[float], resaltados: Iterable[int], titulo: str):
    fig, ax = _fig(titulo)
    marcados = set(resaltados)
    colores = [NARANJA if k in marcados else AZUL for k in ks]
    ax.bar(ks, probs, color=colores, alpha=0.9, width=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("P(X = x)")
    if len(ks) <= 25:
        ax.set_xticks(list(ks))
    ax.bar([], [], color=NARANJA, label="Valores que suman la probabilidad pedida")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def normal(mu: float, sigma: float, lo: float | None, hi: float | None, titulo: str, etiqueta: str = "x"):
    """Curva normal con el área [lo, hi] sombreada (None = infinito)."""
    fig, ax = _fig(titulo)
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
    ys = stats.norm.pdf(xs, mu, sigma)
    ax.plot(xs, ys, color=AZUL, lw=2)
    izq = -np.inf if lo is None else lo
    der = np.inf if hi is None else hi
    mask = (xs >= izq) & (xs <= der)
    ax.fill_between(xs, ys, where=mask, color=NARANJA, alpha=0.45, label="Área = probabilidad")
    for v in (lo, hi):
        if v is not None:
            ax.axvline(v, color=NARANJA, lw=1, ls="--")
    ax.axvline(mu, color=GRIS, lw=1, ls=":")
    ax.set_xlabel(etiqueta)
    ax.set_ylabel("Densidad")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def regresion(x: Sequence[float], y: Sequence[float], b0: float, b1: float, x_pred: float | None):
    fig, ax = _fig("Diagrama de dispersión y recta de regresión")
    ax.scatter(x, y, color=AZUL, zorder=3, label="Datos")
    xs = np.linspace(min(x), max(x), 100)
    if x_pred is not None:
        xs = np.linspace(min(min(x), x_pred), max(max(x), x_pred), 100)
    ax.plot(xs, b0 + b1 * xs, color=NARANJA, lw=2, label=f"ŷ = {b0:.4g} + {b1:.4g}x")
    if x_pred is not None:
        ax.scatter([x_pred], [b0 + b1 * x_pred], color=VERDE, s=70, marker="D", zorder=4,
                   label=f"Predicción en x = {x_pred:g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def intervalo(estimador: float, li: float, ls: float, error_est: float, nombre: str):
    fig, ax = _fig(f"Intervalo de confianza para {nombre}", alto=3.2)
    xs = np.linspace(estimador - 4 * error_est, estimador + 4 * error_est, 400)
    ys = stats.norm.pdf(xs, estimador, error_est)
    ax.plot(xs, ys, color=AZUL, lw=2)
    ax.fill_between(xs, ys, where=(xs >= li) & (xs <= ls), color=NARANJA, alpha=0.35, label="Nivel de confianza")
    ax.axvline(estimador, color=VERDE, lw=2, label=f"Estimador = {estimador:.4g}")
    ax.axvline(li, color=NARANJA, ls="--")
    ax.axvline(ls, color=NARANJA, ls="--")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def prueba(estadistico: float, criticos: Sequence[float], cola: str, gl: float | None, nombre: str):
    """Distribución del estadístico bajo H0 con la región de rechazo sombreada."""
    dist = stats.t(gl) if gl else stats.norm()
    fig, ax = _fig(f"Distribución de {nombre} bajo H₀")
    lim = max(4.0, abs(estadistico) + 0.5)
    xs = np.linspace(-lim, lim, 600)
    ys = dist.pdf(xs)
    ax.plot(xs, ys, color=AZUL, lw=2)
    if cola == "izquierda":
        mask = xs <= criticos[0]
    elif cola == "derecha":
        mask = xs >= criticos[0]
    else:
        mask = (xs <= -abs(criticos[0])) | (xs >= abs(criticos[0]))
    ax.fill_between(xs, ys, where=mask, color=NARANJA, alpha=0.45, label="Región de rechazo")
    ax.axvline(estadistico, color=VERDE, lw=2, label=f"{nombre} calculado = {estadistico:.4f}")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return fig
