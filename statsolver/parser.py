"""Intérprete local de enunciados escritos (no necesita API).

Reconoce frases y notación habituales, por ejemplo:
    binomial n=10 p=0.3 P(X<=2)
    poisson lambda=4 al menos 2
    normal media=70 sigma=8 P(65 < X < 80)
    datos: 4, 8, 15, 16, 23, 42
    x: 1 2 3 4   y: 2 4 5 8   predecir x=6
    intervalo media xbar=52 s=6 n=36 confianza 95%
    hipotesis H0: mu=50 H1: mu>50 xbar=52 s=6 n=36 alfa=0.05
    C(10,3)*0.3^3*0.7^7
"""
from __future__ import annotations

import re
import unicodedata

from .models import Ejercicio

NUM = r"[-+]?\d+(?:\.\d+)?"


def _normalizar(texto: str) -> str:
    reemplazos = {
        "x̄": "xbar", "p̂": "phat", "μ": "mu", "σ": "sigma", "λ": "lambda", "≤": "<=", "≥": ">=",
        "≠": "!=", "−": "-", "×": "*", "·": "*", "²": "^2",
    }
    for a, b in reemplazos.items():
        texto = texto.replace(a, b)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c)).lower()
    # coma decimal: "2,5" -> "2.5" (pero "2, 5" sigue siendo una lista)
    return re.sub(r"(\d),(\d)", r"\1.\2", texto) if ";" in texto or not re.search(r"\d,\s*\d", texto) else texto


def _campo(t: str, *nombres: str) -> float | None:
    patron = rf"(?<![a-z])(?:{'|'.join(nombres)})\s*(?:[:=]|es|de)?\s*({NUM})"
    m = re.search(patron, t)
    return float(m.group(1)) if m else None


def _numeros(t: str) -> list[float]:
    sep = ";" if ";" in t else None
    if sep:
        return [float(x.replace(",", ".")) for x in re.findall(r"-?\d+(?:[.,]\d+)?", t)]
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]


def _porcentaje(t: str, *nombres: str) -> float | None:
    m = re.search(rf"(?:{'|'.join(nombres)})[^\d]{{0,25}}({NUM})\s*%", t)
    if m:
        return float(m.group(1)) / 100
    v = _campo(t, *nombres)
    if v is not None:
        return v / 100 if v > 1 else v
    return None


def _consulta(t: str) -> dict:
    m = re.search(rf"p\s*\(\s*({NUM})\s*(<=|<)\s*x\s*(<=|<)\s*({NUM})\s*\)", t)
    if m:
        return dict(consulta="entre", a=float(m.group(1)), b=float(m.group(4)))
    m = re.search(rf"entre\s+({NUM})\s+y\s+({NUM})", t)
    if m:
        return dict(consulta="entre", a=float(m.group(1)), b=float(m.group(2)))
    m = re.search(rf"(?:xbar|x)\s*(<=|>=|<|>|=)\s*({NUM})", t)
    if m:
        op = {"<=": "menor_igual", ">=": "mayor_igual", "<": "menor", ">": "mayor", "=": "igual"}[m.group(1)]
        return dict(consulta=op, a=float(m.group(2)))
    frases = [
        (r"al menos|como minimo|por lo menos|no menos de", "mayor_igual"),
        (r"a lo mas|a lo sumo|como maximo|no mas de|maximo", "menor_igual"),
        (r"mas de|mayor(?:es)? (?:a|que)", "mayor"),
        (r"menos de|menor(?:es)? (?:a|que)", "menor"),
        (r"exactamente|igual a", "igual"),
    ]
    for patron, op in frases:
        m = re.search(rf"(?:{patron})\s+({NUM})", t)
        if m:
            return dict(consulta=op, a=float(m.group(1)))
    m = re.search(rf"percentil\s*({NUM})", t)
    if m:
        return dict(consulta="percentil", probabilidad=float(m.group(1)) / 100)
    return dict(consulta="ninguna")


def _lista_tras(t: str, etiqueta: str) -> list[float]:
    m = re.search(rf"(?<![a-z]){etiqueta}\s*[:=]\s*([-\d.,;\s]+)", t)
    return _numeros(m.group(1)) if m else []


def interpretar_texto(texto: str) -> Ejercicio:
    original = texto.strip()
    t = _normalizar(original)
    base = dict(transcripcion=original, pregunta="")

    # --- binomial
    if "binomial" in t or re.search(r"\bb\s*\(", t):
        m = re.search(rf"\bb\s*\(\s*({NUM})\s*[,;]\s*({NUM})\s*\)", t)
        n = float(m.group(1)) if m else _campo(t, "n", "ensayos")
        p = float(m.group(2)) if m else _porcentaje(t, "p", "probabilidad de exito", "probabilidad")
        return Ejercicio.vacio(**base, tipo="binomial", n=n, p=p, **_consulta(t))

    # --- Poisson
    if "poisson" in t or "lambda" in t:
        return Ejercicio.vacio(**base, tipo="poisson", lam=_campo(t, "lambda", "promedio", "media"), **_consulta(t))

    # --- prueba de hipótesis
    if re.search(r"hipotesis|\bh0\b|\bho\b|contraste", t):
        prop = bool(re.search(r"proporcion|phat|\bp0\b", t))
        cola = "bilateral"
        m = re.search(r"h[1a]\s*:?\s*(?:mu|p)\s*(!=|<>|<|>)", t)
        if m:
            cola = {"<": "izquierda", ">": "derecha"}.get(m.group(1), "bilateral")
        h0 = re.search(rf"h[0o]\s*:?\s*(?:mu|p)\s*(?:=|<=|>=)\s*({NUM})", t)
        valor0 = float(h0.group(1)) if h0 else _campo(t, "mu0", "p0")
        return Ejercicio.vacio(
            **base, tipo="prueba_hipotesis", parametro="proporcion" if prop else "media", cola=cola,
            mu=None if prop else valor0, p=valor0 if prop else None,
            media_muestral=_campo(t, "xbar", "media muestral", "promedio"),
            desviacion_muestral=_campo(t, "s", "desviacion muestral"),
            sigma=_campo(t, "sigma", "desviacion poblacional"),
            n=_campo(t, "n", "muestra"), proporcion_muestral=_campo(t, "phat", "proporcion muestral"),
            a=_campo(t, "exitos", "x"), alfa=_porcentaje(t, "alfa", "alpha", "significancia"),
            datos=_lista_tras(t, "datos"),
        )

    # --- intervalo de confianza
    if re.search(r"intervalo|confianza", t):
        prop = bool(re.search(r"proporcion|phat", t))
        return Ejercicio.vacio(
            **base, tipo="intervalo_confianza", parametro="proporcion" if prop else "media",
            media_muestral=_campo(t, "xbar", "media muestral", "media", "promedio"),
            desviacion_muestral=_campo(t, "s", "desviacion muestral"),
            sigma=_campo(t, "sigma", "desviacion poblacional"),
            n=_campo(t, "n", "muestra"), proporcion_muestral=_campo(t, "phat", "proporcion muestral", "proporcion"),
            a=_campo(t, "exitos", "x"), confianza=_porcentaje(t, "confianza", "nivel"),
            datos=_lista_tras(t, "datos"),
        )

    # --- normal
    if "normal" in t or re.search(r"\bn\s*\(\s*" + NUM, t):
        m = re.search(rf"\bn\s*\(\s*({NUM})\s*[,;]\s*({NUM})\s*\)", t)
        mu = float(m.group(1)) if m else _campo(t, "mu", "media")
        sigma = float(m.group(2)) if m else _campo(t, "sigma", "desviacion estandar", "desviacion")
        n = _campo(t, "n", "muestra de", "tamano de muestra")
        return Ejercicio.vacio(**base, tipo="normal", mu=mu, sigma=sigma, n=n, **_consulta(t))

    # --- regresión
    xs, ys = _lista_tras(t, "x"), _lista_tras(t, "y")
    if xs and ys:
        pred = re.search(rf"(?:predecir|estimar|prediccion|pronostico)[^\d-]*({NUM})", t)
        return Ejercicio.vacio(**base, tipo="regresion", datos=xs, datos_y=ys,
                               x_prediccion=float(pred.group(1)) if pred else None)

    # --- descriptiva
    datos = _lista_tras(t, "datos")
    solo_numeros = re.fullmatch(r"[-\d.,;\s]+", t)
    if datos or (solo_numeros and len(_numeros(t)) >= 2):
        return Ejercicio.vacio(**base, tipo="descriptiva", datos=datos or _numeros(t),
                               es_poblacion=bool(re.search(r"poblacion", t)))

    # --- fórmula suelta
    return Ejercicio.vacio(**base, tipo="expresion", expresion=original)
