"""Estructuras de datos compartidas: el ejercicio reconocido y su solución."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field

TipoEjercicio = Literal[
    "descriptiva",
    "binomial",
    "poisson",
    "normal",
    "regresion",
    "intervalo_confianza",
    "prueba_hipotesis",
    "expresion",
    "desconocido",
]

Consulta = Literal[
    "igual", "menor_igual", "menor", "mayor_igual", "mayor", "entre", "percentil", "ninguna"
]

TIPOS: dict[str, str] = {
    "descriptiva": "Estadística descriptiva",
    "binomial": "Distribución binomial",
    "poisson": "Distribución de Poisson",
    "normal": "Distribución normal",
    "regresion": "Regresión lineal y correlación",
    "intervalo_confianza": "Intervalo de confianza",
    "prueba_hipotesis": "Prueba de hipótesis",
    "expresion": "Fórmula / expresión matemática",
    "desconocido": "No identificado",
}

CONSULTAS: dict[str, str] = {
    "igual": "P(X = a)",
    "menor_igual": "P(X ≤ a)",
    "menor": "P(X < a)",
    "mayor_igual": "P(X ≥ a)",
    "mayor": "P(X > a)",
    "entre": "P(a ≤ X ≤ b)",
    "percentil": "Valor x tal que P(X ≤ x) = prob",
    "ninguna": "Solo parámetros (media, varianza)",
}


class Ejercicio(BaseModel):
    """Representación normalizada de un ejercicio, venga de la foto o del texto."""

    transcripcion: str = Field(description="Texto completo del enunciado tal como se lee en la imagen.")
    latex: str = Field(description="Fórmula principal en LaTeX, o cadena vacía si no hay.")
    tipo: TipoEjercicio = Field(description="Tipo de ejercicio estadístico.")
    pregunta: str = Field(description="Qué se pide calcular, en una frase corta.")

    datos: List[float] = Field(description="Lista de datos numéricos (muestra, o valores de X en regresión).")
    datos_y: List[float] = Field(description="Valores de Y para regresión; vacío en otro caso.")
    es_poblacion: bool = Field(description="True si los datos son toda la población (varianza con n).")

    n: Optional[float] = Field(description="Número de ensayos (binomial) o tamaño de muestra.")
    p: Optional[float] = Field(description="Probabilidad de éxito (binomial) o proporción poblacional p0 (prueba de proporción).")
    lam: Optional[float] = Field(description="Parámetro lambda (Poisson).")
    mu: Optional[float] = Field(description="Media poblacional (normal) o mu0 de la hipótesis nula.")
    sigma: Optional[float] = Field(description="Desviación estándar poblacional, si se conoce.")
    media_muestral: Optional[float] = Field(description="Media muestral x̄, si se da.")
    desviacion_muestral: Optional[float] = Field(description="Desviación estándar muestral s, si se da.")
    proporcion_muestral: Optional[float] = Field(description="Proporción muestral p̂, si se da.")

    a: Optional[float] = Field(description="Valor a de la consulta de probabilidad (o k éxitos).")
    b: Optional[float] = Field(description="Valor b (límite superior) cuando la consulta es 'entre'.")
    consulta: Consulta = Field(description="Tipo de probabilidad pedida.")
    probabilidad: Optional[float] = Field(description="Probabilidad dada cuando se pide un percentil.")

    confianza: Optional[float] = Field(description="Nivel de confianza como decimal, p. ej. 0.95.")
    alfa: Optional[float] = Field(description="Nivel de significancia, p. ej. 0.05.")
    cola: Literal["bilateral", "izquierda", "derecha"] = Field(description="Tipo de prueba (H1: ≠, <, >).")
    parametro: Literal["media", "proporcion"] = Field(description="Parámetro del intervalo o la prueba.")

    x_prediccion: Optional[float] = Field(description="Valor de x para predecir y en regresión.")
    expresion: str = Field(description="Expresión en sintaxis Python/SymPy a evaluar (tipo 'expresion').")

    @classmethod
    def vacio(cls, **kwargs: Any) -> "Ejercicio":
        base: dict[str, Any] = dict(
            transcripcion="", latex="", tipo="desconocido", pregunta="",
            datos=[], datos_y=[], es_poblacion=False,
            n=None, p=None, lam=None, mu=None, sigma=None,
            media_muestral=None, desviacion_muestral=None, proporcion_muestral=None,
            a=None, b=None, consulta="ninguna", probabilidad=None,
            confianza=None, alfa=None, cola="bilateral", parametro="media",
            x_prediccion=None, expresion="",
        )
        base.update(kwargs)
        return cls(**base)


@dataclass
class Paso:
    titulo: str
    texto: str = ""
    latex: List[str] = field(default_factory=list)
    tabla: Any = None  # pandas.DataFrame opcional


@dataclass
class Solucion:
    titulo: str
    pasos: List[Paso] = field(default_factory=list)
    resultados: dict[str, Any] = field(default_factory=dict)
    figuras: List[Any] = field(default_factory=list)  # matplotlib Figure
    conclusion: str = ""
    advertencias: List[str] = field(default_factory=list)

    def paso(self, titulo: str, texto: str = "", *latex: str, tabla: Any = None) -> None:
        self.pasos.append(Paso(titulo, texto, list(latex), tabla))
