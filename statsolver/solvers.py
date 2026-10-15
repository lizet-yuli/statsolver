"""Motores de solución paso a paso.

Todo el cálculo es determinista (SymPy para la aritmética exacta, SciPy para
cuantiles y funciones de distribución): el reconocimiento de la imagen solo
extrae los datos, nunca decide el resultado.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from tokenize import TokenError
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd
import sympy as sp
from scipy import stats
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from . import plots
from .models import Ejercicio, Solucion


# ---------------------------------------------------------------- utilidades

def R(x: float) -> sp.Rational:
    """Convierte un decimal a racional exacto (0.3 -> 3/10)."""
    return sp.Rational(str(round(float(x), 10)))


def fx(x, d: int = 4) -> str:
    """Formatea un número para LaTeX."""
    x = float(x)
    if math.isfinite(x) and x == int(x) and abs(x) < 1e12:
        return str(int(x))
    if x != 0 and abs(x) < 1e-4:
        mant, exp = f"{x:.4e}".split("e")
        return rf"{mant} \times 10^{{{int(exp)}}}"
    return f"{x:.{d}f}".rstrip("0").rstrip(".")


def ft(x, d: int = 4) -> str:
    """Formatea un número para texto plano."""
    x = float(x)
    if math.isfinite(x) and x == int(x) and abs(x) < 1e12:
        return str(int(x))
    if x != 0 and abs(x) < 1e-4:
        return f"{x:.4e}"
    return f"{x:.{d}f}".rstrip("0").rstrip(".")


def _lista(vals: Sequence, maximo: int = 15) -> str:
    vals = [fx(v) for v in vals]
    if len(vals) > maximo:
        vals = vals[:10] + [r"\ldots"] + vals[-3:]
    return r"\{" + r",\ ".join(vals) + r"\}"


def _suma(vals: Sequence, maximo: int = 12) -> str:
    vals = [fx(v) if float(v) >= 0 else f"({fx(v)})" for v in vals]
    if len(vals) > maximo:
        vals = vals[:6] + [r"\cdots"] + vals[-2:]
    return " + ".join(vals)


def _requerir(valor, nombre: str):
    if valor is None:
        raise ValueError(f"Falta el dato: {nombre}.")
    return valor


# ------------------------------------------------------- estadística descriptiva

def _tabla_frecuencias(datos: List[float]):
    n = len(datos)
    unicos = sorted(set(datos))
    if len(unicos) <= 12:
        conteo = Counter(datos)
        f = np.array([conteo[u] for u in unicos])
        df = pd.DataFrame({"xᵢ": [ft(u) for u in unicos], "fᵢ": f})
        texto = "Cada valor distinto con su frecuencia absoluta (fᵢ), relativa (hᵢ) y acumuladas (Fᵢ, Hᵢ)."
    else:
        k = math.ceil(1 + 3.322 * math.log10(n))
        bordes = np.linspace(min(datos), max(datos), k + 1)
        f, _ = np.histogram(datos, bins=bordes)
        etiquetas = [
            f"[{ft(bordes[i], 2)}, {ft(bordes[i + 1], 2)}{']' if i == k - 1 else ')'}"
            for i in range(k)
        ]
        marcas = [(bordes[i] + bordes[i + 1]) / 2 for i in range(k)]
        df = pd.DataFrame({"Clase": etiquetas, "Marca xᵢ": [ft(m, 3) for m in marcas], "fᵢ": f})
        texto = (
            f"Como hay muchos valores distintos se agrupan en clases. Regla de Sturges: "
            f"k = 1 + 3.322·log₁₀({n}) ≈ {k} clases de amplitud "
            f"{ft((max(datos) - min(datos)) / k, 3)}."
        )
    df["hᵢ"] = np.round(f / n, 4)
    df["Fᵢ"] = np.cumsum(f)
    df["Hᵢ"] = np.round(np.cumsum(f) / n, 4)
    return df, texto


def resolver_descriptiva(e: Ejercicio) -> Solucion:
    if len(e.datos) < 2:
        raise ValueError("Se necesitan al menos 2 datos para la estadística descriptiva.")
    datos = [float(v) for v in e.datos]
    xs = [R(v) for v in datos]
    n = len(xs)
    pob = e.es_poblacion
    m = r"\mu" if pob else r"\bar{x}"
    sol = Solucion("Estadística descriptiva" + (" (población)" if pob else " (muestra)"))

    orden = sorted(xs)
    sol.paso("Ordenar los datos", f"Hay **n = {n}** datos. Ordenados de menor a mayor:",
             r"\text{Datos: } " + _lista(orden))

    tabla, texto = _tabla_frecuencias(datos)
    sol.paso("Tabla de frecuencias", texto, tabla=tabla)

    suma = sum(xs)
    media = suma / n
    sol.paso("Media aritmética", "Se suman todos los datos y se divide entre la cantidad de datos.",
             rf"{m} = \frac{{\sum x_i}}{{n}} = \frac{{{_suma(xs)}}}{{{n}}}",
             rf"{m} = \frac{{{fx(suma)}}}{{{n}}} = {fx(media)}")

    if n % 2:
        pos = (n + 1) // 2
        mediana = orden[pos - 1]
        sol.paso("Mediana", f"n = {n} es impar: la mediana es el dato central, en la posición (n+1)/2 = {pos}.",
                 rf"Me = x_{{({pos})}} = {fx(mediana)}")
    else:
        p1 = n // 2
        mediana = (orden[p1 - 1] + orden[p1]) / 2
        sol.paso("Mediana", f"n = {n} es par: la mediana es el promedio de los datos en las posiciones {p1} y {p1 + 1}.",
                 rf"Me = \frac{{x_{{({p1})}} + x_{{({p1 + 1})}}}}{{2}} = \frac{{{fx(orden[p1 - 1])} + {fx(orden[p1])}}}{{2}} = {fx(mediana)}")

    conteo = Counter(datos)
    fmax = max(conteo.values())
    if fmax == 1:
        moda_txt = "Ningún valor se repite: la distribución es **amodal** (no tiene moda)."
        modas: list[float] = []
    else:
        modas = sorted(k for k, v in conteo.items() if v == fmax)
        clase = {1: "unimodal", 2: "bimodal"}.get(len(modas), "multimodal")
        moda_txt = f"El valor (o valores) que más se repite(n), {fmax} veces. Distribución {clase}."
    sol.paso("Moda", moda_txt, *([rf"Mo = {', '.join(fx(v) for v in modas)}"] if modas else []))

    rango = orden[-1] - orden[0]
    sol.paso("Rango", "Diferencia entre el dato mayor y el menor.",
             rf"R = x_{{máx}} - x_{{mín}} = {fx(orden[-1])} - {fx(orden[0])} = {fx(rango)}")

    desv = [x - media for x in xs]
    cuad = [d ** 2 for d in desv]
    ss = sum(cuad)
    den = n if pob else n - 1
    var = ss / den
    tabla_var = pd.DataFrame({
        "xᵢ": [ft(x) for x in datos],
        "xᵢ − media": [ft(d) for d in desv],
        "(xᵢ − media)²": [ft(c) for c in cuad],
    })
    tabla_var.loc[len(tabla_var)] = ["Σ", ft(sum(desv)), ft(ss)]
    s2, s = (r"\sigma^2", r"\sigma") if pob else ("s^2", "s")
    explicacion = (
        "Se calcula la desviación de cada dato respecto a la media, se eleva al cuadrado y se suman. "
        + ("Como son datos de **población** se divide entre n." if pob
           else "Como es una **muestra** se divide entre n − 1 (corrección de Bessel).")
    )
    sol.paso("Varianza", explicacion,
             rf"{s2} = \frac{{\sum (x_i - {m})^2}}{{{'n' if pob else 'n-1'}}} = \frac{{{fx(ss)}}}{{{den}}} = {fx(var)}",
             tabla=tabla_var if n <= 40 else None)

    desv_est = sp.sqrt(var)
    sol.paso("Desviación estándar", "Es la raíz cuadrada de la varianza; está en las mismas unidades que los datos.",
             rf"{s} = \sqrt{{{s2}}} = \sqrt{{{fx(var)}}} = {fx(desv_est)}")

    cv = None
    if media != 0:
        cv = desv_est / abs(media) * 100
        nivel = "homogéneos" if float(cv) < 15 else ("moderadamente dispersos" if float(cv) < 30 else "heterogéneos")
        sol.paso("Coeficiente de variación", f"Mide la dispersión relativa a la media. Los datos son **{nivel}**.",
                 rf"CV = \frac{{{s}}}{{|{m}|}} \times 100\% = \frac{{{fx(desv_est)}}}{{{fx(abs(media))}}} \times 100\% = {fx(cv, 2)}\%")

    q1, q2, q3 = np.percentile(datos, [25, 50, 75], method="weibull")
    ric = q3 - q1
    li, ls = q1 - 1.5 * ric, q3 + 1.5 * ric
    atipicos = [v for v in datos if v < li or v > ls]
    sol.paso(
        "Cuartiles y datos atípicos",
        "Se usa la posición (n+1)·k/4 sobre los datos ordenados, interpolando cuando no es entera. "
        + (f"Datos atípicos: **{', '.join(ft(a) for a in atipicos)}**." if atipicos else "No hay datos atípicos."),
        rf"\text{{Pos}}(Q_1) = \frac{{{n}+1}}{{4}} = {fx((n + 1) / 4)},\quad \text{{Pos}}(Q_3) = \frac{{3({n}+1)}}{{4}} = {fx(3 * (n + 1) / 4)}",
        rf"Q_1 = {fx(q1)},\quad Q_2 = {fx(q2)},\quad Q_3 = {fx(q3)},\quad RIC = Q_3 - Q_1 = {fx(ric)}",
        rf"\text{{Límites: }} [Q_1 - 1.5\,RIC,\ Q_3 + 1.5\,RIC] = [{fx(li)},\ {fx(ls)}]",
    )

    sol.resultados = {
        "n": n, "Media": float(media), "Mediana": float(mediana),
        "Moda": ", ".join(ft(v) for v in modas) if modas else "—",
        "Rango": float(rango), "Varianza": float(var), "Desv. estándar": float(desv_est),
        "CV (%)": float(cv) if cv is not None else "—", "Q1": float(q1), "Q3": float(q3),
    }
    sol.figuras = [plots.histograma(datos, float(media), float(mediana)), plots.caja(datos)]
    sol.conclusion = (
        f"La media es {ft(media)} con una desviación estándar de {ft(desv_est)}; "
        f"la mediana es {ft(mediana)}."
    )
    return sol


# ------------------------------------------------ distribuciones discretas

def _evento(consulta: str, a, b, var: str = "X") -> str:
    a = fx(a) if a is not None else "a"
    b = fx(b) if b is not None else "b"
    return {
        "igual": f"{var} = {a}", "menor_igual": rf"{var} \le {a}", "menor": f"{var} < {a}",
        "mayor_igual": rf"{var} \ge {a}", "mayor": f"{var} > {a}", "entre": rf"{a} \le {var} \le {b}",
    }[consulta]


def _probabilidad_discreta(sol: Solucion, consulta: str, a: int, b: Optional[int], maximo: Optional[int],
                           pmf: Callable[[int], sp.Expr], pmf_latex: Callable[[int], str]):
    """Calcula P(evento) sumando términos, usando el complemento si es más corto."""
    tope = maximo if maximo is not None else None
    if consulta == "igual":
        directos = [a]
    elif consulta == "menor_igual":
        directos = list(range(0, a + 1))
    elif consulta == "menor":
        directos = list(range(0, a))
    elif consulta == "entre":
        directos = list(range(a, _requerir(b, "límite superior b") + 1))
    elif consulta in ("mayor_igual", "mayor"):
        inicio = a if consulta == "mayor_igual" else a + 1
        directos = list(range(inicio, tope + 1)) if tope is not None else None
    else:
        raise ValueError(f"Consulta no soportada: {consulta}")

    if tope is not None:
        directos = [k for k in directos if 0 <= k <= tope]

    evento = _evento(consulta, a, b)
    complemento = None
    if consulta in ("mayor_igual", "mayor"):
        corte = a - 1 if consulta == "mayor_igual" else a
        complemento = list(range(0, corte + 1))
    usar_comp = complemento is not None and (directos is None or len(complemento) < len(directos))
    sumar = complemento if usar_comp else directos

    if usar_comp:
        corte = complemento[-1] if complemento else -1
        sol.paso("Usar el complemento",
                 "Sumar todos los valores pedidos sería más largo (o infinito), así que se usa la regla del complemento.",
                 rf"P({evento}) = 1 - P(X \le {corte})")

    terminos = []
    for k in sumar:
        valor = pmf(k)
        terminos.append(valor)
    lineas = [rf"P(X={k}) = {pmf_latex(k)} = {fx(sp.N(v), 6)}" for k, v in zip(sumar[:10], terminos[:10])]
    if len(sumar) > 10:
        lineas.append(rf"\vdots \quad \text{{({len(sumar) - 10} términos más)}}")
    total = sp.N(sum(terminos), 15) if terminos else sp.Float(0)
    sol.paso("Calcular cada probabilidad", "Se aplica la función de probabilidad a cada valor de X:", *lineas)

    if usar_comp:
        resultado = 1 - total
        sol.paso("Resultado", "", rf"P(X \le {sumar[-1] if sumar else -1}) = {fx(total, 6)}",
                 rf"P({evento}) = 1 - {fx(total, 6)} = {fx(resultado, 6)}")
    else:
        resultado = total
        partes = " + ".join(fx(sp.N(t), 6) for t in terminos[:6]) + (r" + \cdots" if len(terminos) > 6 else "")
        sol.paso("Resultado", "Se suman las probabilidades." if len(terminos) > 1 else "",
                 rf"P({evento}) = {partes if len(terminos) > 1 else fx(total, 6)}" + (rf" = {fx(total, 6)}" if len(terminos) > 1 else ""))
    return float(resultado), evento, (directos, complemento, usar_comp)


def _resaltados(info, rango_graf) -> set[int]:
    directos, complemento, _ = info
    if directos is not None:
        return set(directos)
    return {k for k in rango_graf if k not in set(complemento)}


def resolver_binomial(e: Ejercicio) -> Solucion:
    n = int(_requerir(e.n, "número de ensayos n"))
    p = R(_requerir(e.p, "probabilidad de éxito p"))
    if not (0 <= p <= 1) or n < 1:
        raise ValueError("Se requiere n ≥ 1 y 0 ≤ p ≤ 1.")
    q = 1 - p
    sol = Solucion("Distribución binomial")
    sol.paso("Identificar el modelo",
             "X cuenta el número de éxitos en n ensayos independientes, cada uno con la misma probabilidad de éxito p.",
             rf"X \sim B(n = {n},\ p = {fx(p)}),\qquad q = 1 - p = {fx(q)}",
             r"P(X = k) = \binom{n}{k}\, p^{k}\, q^{\,n-k}")

    pmf = lambda k: sp.binomial(n, k) * p ** k * q ** (n - k)  # noqa: E731
    pmf_latex = lambda k: rf"\binom{{{n}}}{{{k}}}({fx(p)})^{{{k}}}({fx(q)})^{{{n - k}}}"  # noqa: E731

    rango_graf = list(range(0, n + 1)) if n <= 60 else list(range(max(0, int(n * p - 4 * math.sqrt(n * p * q))), min(n, int(n * p + 4 * math.sqrt(n * p * q))) + 1))
    resaltados: set[int] = set()
    if e.consulta == "percentil":
        objetivo = _requerir(e.probabilidad, "probabilidad acumulada")
        acum, k = 0.0, 0
        while k <= n:
            acum += float(pmf(k))
            if acum >= objetivo - 1e-12:
                break
            k += 1
        sol.paso("Percentil", "Se acumulan probabilidades hasta alcanzar la probabilidad pedida.",
                 rf"P(X \le {k}) = {fx(acum, 6)} \ge {fx(objetivo)}")
        sol.resultados["Percentil"] = k
        resaltados = set(range(0, k + 1))
    elif e.consulta != "ninguna":
        a = int(round(_requerir(e.a, "valor a de la consulta")))
        b = int(round(e.b)) if e.b is not None else None
        res, evento, info = _probabilidad_discreta(sol, e.consulta, a, b, n, pmf, pmf_latex)
        sol.resultados[f"P({evento.replace(chr(92) + 'le', '≤').replace(chr(92) + 'ge', '≥')})"] = res
        sol.conclusion = f"La probabilidad pedida es {ft(res, 6)} (≈ {ft(res * 100, 2)} %)."
        resaltados = _resaltados(info, rango_graf)

    media, var = n * p, n * p * q
    sol.paso("Media, varianza y desviación estándar", "",
             rf"\mu = np = {n}({fx(p)}) = {fx(media)}",
             rf"\sigma^2 = npq = {n}({fx(p)})({fx(q)}) = {fx(var)}",
             rf"\sigma = \sqrt{{{fx(var)}}} = {fx(sp.sqrt(var))}")
    sol.resultados.update({"Media μ": float(media), "Varianza σ²": float(var), "Desv. σ": float(sp.sqrt(var))})
    sol.figuras = [plots.discreta(rango_graf, [float(pmf(k)) for k in rango_graf], resaltados,
                                  f"Función de probabilidad B({n}, {ft(p)})")]
    return sol


def resolver_poisson(e: Ejercicio) -> Solucion:
    lam = R(_requerir(e.lam, "λ (promedio de ocurrencias)"))
    if lam <= 0:
        raise ValueError("λ debe ser positivo.")
    sol = Solucion("Distribución de Poisson")
    sol.paso("Identificar el modelo",
             "X cuenta el número de ocurrencias en un intervalo (tiempo, área, etc.) con una tasa promedio λ.",
             rf"X \sim \text{{Poisson}}(\lambda = {fx(lam)})",
             r"P(X = k) = \frac{e^{-\lambda}\,\lambda^{k}}{k!}")
    pmf = lambda k: sp.exp(-lam) * lam ** k / sp.factorial(k)  # noqa: E731
    pmf_latex = lambda k: rf"\frac{{e^{{-{fx(lam)}}}({fx(lam)})^{{{k}}}}}{{{k}!}}"  # noqa: E731

    tope = int(float(lam) + 5 * math.sqrt(float(lam)) + 5)
    rango_graf = list(range(0, tope + 1))
    resaltados: set[int] = set()
    if e.consulta == "percentil":
        objetivo = _requerir(e.probabilidad, "probabilidad acumulada")
        acum, k = 0.0, 0
        while True:
            acum += float(pmf(k))
            if acum >= objetivo - 1e-12 or k > 10_000:
                break
            k += 1
        sol.paso("Percentil", "Se acumulan probabilidades hasta alcanzar la probabilidad pedida.",
                 rf"P(X \le {k}) = {fx(acum, 6)} \ge {fx(objetivo)}")
        sol.resultados["Percentil"] = k
        resaltados = set(range(0, k + 1))
    elif e.consulta != "ninguna":
        a = int(round(_requerir(e.a, "valor a de la consulta")))
        b = int(round(e.b)) if e.b is not None else None
        res, evento, info = _probabilidad_discreta(sol, e.consulta, a, b, None, pmf, pmf_latex)
        sol.resultados[f"P({evento.replace(chr(92) + 'le', '≤').replace(chr(92) + 'ge', '≥')})"] = res
        sol.conclusion = f"La probabilidad pedida es {ft(res, 6)} (≈ {ft(res * 100, 2)} %)."
        resaltados = _resaltados(info, rango_graf)

    sol.paso("Media y varianza", "En la distribución de Poisson la media y la varianza son iguales a λ.",
             rf"\mu = \sigma^2 = \lambda = {fx(lam)},\qquad \sigma = \sqrt{{{fx(lam)}}} = {fx(sp.sqrt(lam))}")
    sol.resultados.update({"Media μ": float(lam), "Varianza σ²": float(lam)})
    sol.figuras = [plots.discreta(rango_graf, [float(pmf(k)) for k in rango_graf], resaltados,
                                  f"Función de probabilidad Poisson(λ = {ft(lam)})")]
    return sol


# --------------------------------------------------------------------- normal

def resolver_normal(e: Ejercicio) -> Solucion:
    mu = float(_requerir(e.mu, "media μ"))
    sigma = float(_requerir(e.sigma, "desviación estándar σ"))
    if sigma <= 0:
        raise ValueError("σ debe ser positiva.")
    sol = Solucion("Distribución normal")
    usa_media = e.n is not None and e.n > 1
    var = r"\bar{X}" if usa_media else "X"
    se = sigma / math.sqrt(e.n) if usa_media else sigma

    lineas = [rf"X \sim N(\mu = {fx(mu)},\ \sigma = {fx(sigma)})"]
    texto = "La variable sigue una distribución normal con media μ y desviación estándar σ."
    if usa_media:
        texto += (" Se pregunta por la **media de una muestra** de tamaño n, así que por el Teorema del "
                  "Límite Central se usa el error estándar σ/√n.")
        lineas.append(rf"\sigma_{{\bar{{X}}}} = \frac{{\sigma}}{{\sqrt{{n}}}} = \frac{{{fx(sigma)}}}{{\sqrt{{{fx(e.n)}}}}} = {fx(se)}")
    sol.paso("Identificar el modelo", texto, *lineas)
    sd_txt = r"\sigma_{\bar{X}}" if usa_media else r"\sigma"

    if e.consulta == "percentil":
        prob = float(_requerir(e.probabilidad, "probabilidad acumulada"))
        z = stats.norm.ppf(prob)
        x = mu + z * se
        sol.paso("Buscar z en la tabla", f"Se busca el valor z cuya área acumulada a la izquierda es {ft(prob)}.",
                 rf"\Phi(z) = {fx(prob)} \;\Rightarrow\; z = {fx(z)}")
        sol.paso("Despejar x", "Se despeja x de la fórmula de estandarización.",
                 rf"x = \mu + z\,{sd_txt} = {fx(mu)} + ({fx(z)})({fx(se)}) = {fx(x)}")
        sol.resultados = {"z": z, "x": x}
        sol.conclusion = f"El valor buscado es x ≈ {ft(x)}."
        sol.figuras = [plots.normal(mu, se, None, x, f"N({ft(mu)}, {ft(se)})", "x")]
        return sol

    if e.consulta in ("ninguna",):
        raise ValueError("Indica qué probabilidad se pide (por ejemplo P(X < a)).")
    if e.consulta == "igual":
        sol.paso("Resultado", "En una distribución continua la probabilidad de un valor exacto es 0.",
                 rf"P({var} = {fx(e.a)}) = 0")
        sol.resultados = {"Probabilidad": 0.0}
        return sol

    a = float(_requerir(e.a, "valor a"))
    if e.consulta in ("menor", "menor_igual"):
        lo, hi = None, a
    elif e.consulta in ("mayor", "mayor_igual"):
        lo, hi = a, None
    else:
        lo, hi = a, float(_requerir(e.b, "valor b"))
    evento = _evento(e.consulta, a, e.b, var)

    zs = {}
    est = []
    for x in (lo, hi):
        if x is not None:
            z = (x - mu) / se
            zs[x] = z
            est.append(rf"z_{{{fx(x)}}} = \frac{{x - \mu}}{{{sd_txt}}} = \frac{{{fx(x)} - {fx(mu)}}}{{{fx(se)}}} = {fx(z)}")
    sol.paso("Estandarizar", "Se transforma a la normal estándar Z ~ N(0, 1).", *est)

    phi = {x: stats.norm.cdf(z) for x, z in zs.items()}
    tabla_txt = "; ".join(f"Φ({z:.2f}) ≈ {stats.norm.cdf(round(z, 2)):.4f}" for z in zs.values())
    sol.paso("Buscar en la tabla normal", f"Con tabla (z redondeado a 2 decimales): {tabla_txt}.",
             *[rf"\Phi({fx(z)}) = {fx(phi[x], 6)}" for x, z in zs.items()])

    if lo is None:
        res = phi[hi]
        linea = rf"P({evento}) = P(Z \le {fx(zs[hi])}) = \Phi({fx(zs[hi])}) = {fx(res, 6)}"
    elif hi is None:
        res = 1 - phi[lo]
        linea = rf"P({evento}) = 1 - \Phi({fx(zs[lo])}) = 1 - {fx(phi[lo], 6)} = {fx(res, 6)}"
    else:
        res = phi[hi] - phi[lo]
        linea = rf"P({evento}) = \Phi({fx(zs[hi])}) - \Phi({fx(zs[lo])}) = {fx(phi[hi], 6)} - {fx(phi[lo], 6)} = {fx(res, 6)}"
    nota = "En una variable continua, ≤ y < dan la misma probabilidad." if e.consulta in ("menor_igual", "mayor_igual") else ""
    sol.paso("Calcular la probabilidad", nota, linea)
    sol.resultados = {**{f"z ({ft(x)})": z for x, z in zs.items()}, "Probabilidad": float(res)}
    sol.conclusion = f"La probabilidad pedida es {ft(res, 6)} (≈ {ft(res * 100, 2)} %)."
    sol.figuras = [plots.normal(mu, se, lo, hi, f"N(μ = {ft(mu)}, σ = {ft(se)})", "x̄" if usa_media else "x")]
    return sol


# ------------------------------------------------------------------ regresión

def resolver_regresion(e: Ejercicio) -> Solucion:
    if len(e.datos) != len(e.datos_y) or len(e.datos) < 3:
        raise ValueError("La regresión necesita listas X e Y del mismo tamaño (al menos 3 pares).")
    x = [R(v) for v in e.datos]
    y = [R(v) for v in e.datos_y]
    n = len(x)
    sx, sy = sum(x), sum(y)
    sxy = sum(a * b for a, b in zip(x, y))
    sxx = sum(a ** 2 for a in x)
    syy = sum(b ** 2 for b in y)
    sol = Solucion("Regresión lineal simple y correlación")

    tabla = pd.DataFrame({
        "x": [ft(v) for v in x], "y": [ft(v) for v in y],
        "x·y": [ft(a * b) for a, b in zip(x, y)], "x²": [ft(a ** 2) for a in x], "y²": [ft(b ** 2) for b in y],
    })
    tabla.loc[len(tabla)] = ["Σ " + ft(sx), ft(sy), ft(sxy), ft(sxx), ft(syy)]
    sol.paso("Tabla de sumas", f"Con n = {n} pares se calculan las sumas necesarias.",
             rf"\sum x = {fx(sx)},\ \sum y = {fx(sy)},\ \sum xy = {fx(sxy)},\ \sum x^2 = {fx(sxx)},\ \sum y^2 = {fx(syy)}",
             tabla=tabla)

    den_x = n * sxx - sx ** 2
    den_y = n * syy - sy ** 2
    if den_x == 0:
        raise ValueError("Todos los valores de x son iguales: no se puede ajustar una recta.")
    num = n * sxy - sx * sy
    b1 = num / den_x
    b0 = sy / n - b1 * sx / n
    sol.paso("Pendiente b₁", "Mide cuánto cambia y, en promedio, por cada unidad que aumenta x.",
             r"b_1 = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}",
             rf"b_1 = \frac{{{n}({fx(sxy)}) - ({fx(sx)})({fx(sy)})}}{{{n}({fx(sxx)}) - ({fx(sx)})^2}} = \frac{{{fx(num)}}}{{{fx(den_x)}}} = {fx(b1)}")
    sol.paso("Intercepto b₀", "Es el valor estimado de y cuando x = 0.",
             r"b_0 = \bar{y} - b_1\bar{x}",
             rf"b_0 = {fx(sy / n)} - ({fx(b1)})({fx(sx / n)}) = {fx(b0)}")
    sol.paso("Ecuación de la recta", "", rf"\hat{{y}} = {fx(b0)} {'+' if b1 >= 0 else '-'} {fx(abs(b1))}\,x")

    r = None
    if den_y != 0:
        r = num / sp.sqrt(den_x * den_y)
        rv = float(r)
        fuerza = "muy fuerte" if abs(rv) >= 0.9 else "fuerte" if abs(rv) >= 0.7 else "moderada" if abs(rv) >= 0.4 else "débil"
        signo = "positiva" if rv > 0 else "negativa"
        sol.paso("Coeficiente de correlación de Pearson", f"Correlación lineal **{fuerza} {signo}**.",
                 r"r = \frac{n\sum xy - \sum x\sum y}{\sqrt{[n\sum x^2 - (\sum x)^2][n\sum y^2 - (\sum y)^2]}}",
                 rf"r = \frac{{{fx(num)}}}{{\sqrt{{({fx(den_x)})({fx(den_y)})}}}} = {fx(r)}")
        sol.paso("Coeficiente de determinación",
                 f"El {ft(rv ** 2 * 100, 2)} % de la variación de y se explica por la relación lineal con x.",
                 rf"r^2 = ({fx(r)})^2 = {fx(rv ** 2)}")

    xp = e.x_prediccion
    if xp is not None:
        yp = b0 + b1 * R(xp)
        sol.paso("Predicción", f"Se reemplaza x = {ft(xp)} en la recta.",
                 rf"\hat{{y}}({fx(xp)}) = {fx(b0)} + ({fx(b1)})({fx(xp)}) = {fx(yp)}")
        sol.resultados["ŷ predicho"] = float(yp)

    sol.resultados = {"b₁ (pendiente)": float(b1), "b₀ (intercepto)": float(b0),
                      **({"r": float(r), "r²": float(r) ** 2} if r is not None else {}), **sol.resultados}
    sol.conclusion = f"Recta de regresión: ŷ = {ft(b0)} + {ft(b1)}·x."
    sol.figuras = [plots.regresion([float(v) for v in x], [float(v) for v in y], float(b0), float(b1), xp)]
    return sol


# ------------------------------------------------------- inferencia (IC y PH)

def _resumen_muestral(sol: Solucion, e: Ejercicio):
    """Devuelve (x̄, s, n) a partir de los datos crudos o de los valores dados."""
    if len(e.datos) >= 2:
        d = np.array(e.datos, dtype=float)
        n = len(d)
        xbar, s = d.mean(), d.std(ddof=1)
        sol.paso("Resumen de la muestra", "Se calculan la media y la desviación estándar muestral de los datos.",
                 rf"n = {n},\quad \bar{{x}} = \frac{{\sum x_i}}{{n}} = {fx(xbar)},\quad s = \sqrt{{\frac{{\sum (x_i-\bar{{x}})^2}}{{n-1}}}} = {fx(s)}")
        return xbar, s, n
    xbar = float(_requerir(e.media_muestral, "media muestral x̄"))
    n = int(_requerir(e.n, "tamaño de muestra n"))
    s = e.desviacion_muestral
    return xbar, (float(s) if s is not None else None), n


def resolver_intervalo(e: Ejercicio) -> Solucion:
    conf = e.confianza or (1 - e.alfa if e.alfa else 0.95)
    if conf > 1:
        conf /= 100
    alfa = 1 - conf
    sol = Solucion(f"Intervalo de confianza del {ft(conf * 100, 2)} % para la {e.parametro}")

    if e.parametro == "proporcion":
        n = int(_requerir(e.n, "tamaño de muestra n"))
        if e.proporcion_muestral is not None:
            ph = float(e.proporcion_muestral)
            sol.paso("Proporción muestral", "", rf"\hat{{p}} = {fx(ph)},\quad n = {n}")
        else:
            x = float(_requerir(e.a, "número de éxitos"))
            ph = x / n
            sol.paso("Proporción muestral", "Éxitos entre el tamaño de muestra.",
                     rf"\hat{{p}} = \frac{{x}}{{n}} = \frac{{{fx(x)}}}{{{n}}} = {fx(ph)}")
        z = stats.norm.ppf(1 - alfa / 2)
        sol.paso("Valor crítico", f"α = {ft(alfa)}, se busca z con área {ft(1 - alfa / 2)} a la izquierda.",
                 rf"z_{{\alpha/2}} = z_{{{fx(alfa / 2)}}} = {fx(z)}")
        ee = math.sqrt(ph * (1 - ph) / n)
        E = z * ee
        sol.paso("Error estándar y margen de error", "",
                 rf"EE = \sqrt{{\frac{{\hat{{p}}(1-\hat{{p}})}}{{n}}}} = \sqrt{{\frac{{{fx(ph)}({fx(1 - ph)})}}{{{n}}}}} = {fx(ee)}",
                 rf"E = z_{{\alpha/2}}\cdot EE = {fx(z)}({fx(ee)}) = {fx(E)}")
        li, ls = ph - E, ph + E
        sol.paso("Intervalo", "", rf"IC = \hat{{p}} \pm E = {fx(ph)} \pm {fx(E)} = [{fx(li)},\ {fx(ls)}]")
        if n * ph < 5 or n * (1 - ph) < 5:
            sol.advertencias.append("n·p̂ o n·(1−p̂) es menor que 5: la aproximación normal puede no ser adecuada.")
        sol.resultados = {"p̂": ph, "z crítico": z, "Margen de error": E, "Límite inferior": li, "Límite superior": ls}
        sol.conclusion = (f"Con un {ft(conf * 100, 2)} % de confianza, la proporción poblacional está entre "
                          f"{ft(li)} y {ft(ls)}.")
        sol.figuras = [plots.intervalo(ph, li, ls, ee, "p")]
        return sol

    xbar, s, n = _resumen_muestral(sol, e)
    if e.sigma is not None:
        sd, crit = float(e.sigma), stats.norm.ppf(1 - alfa / 2)
        sol.paso("Valor crítico (σ conocida → Z)", f"Como se conoce σ se usa la normal estándar. α = {ft(alfa)}.",
                 rf"z_{{\alpha/2}} = {fx(crit)}")
        simb, nombre = r"\sigma", "z"
    else:
        sd = _requerir(s, "desviación estándar (s o σ)")
        gl = n - 1
        crit = stats.t.ppf(1 - alfa / 2, gl)
        sol.paso("Valor crítico (σ desconocida → t de Student)",
                 f"No se conoce σ, se usa s y la distribución t con n − 1 = {gl} grados de libertad. α = {ft(alfa)}.",
                 rf"t_{{\alpha/2,\,{gl}}} = {fx(crit)}")
        simb, nombre = "s", "t"
    ee = sd / math.sqrt(n)
    E = crit * ee
    sol.paso("Error estándar y margen de error", "",
             rf"EE = \frac{{{simb}}}{{\sqrt{{n}}}} = \frac{{{fx(sd)}}}{{\sqrt{{{n}}}}} = {fx(ee)}",
             rf"E = {nombre}\cdot EE = {fx(crit)}({fx(ee)}) = {fx(E)}")
    li, ls = xbar - E, xbar + E
    sol.paso("Intervalo", "", rf"IC = \bar{{x}} \pm E = {fx(xbar)} \pm {fx(E)} = [{fx(li)},\ {fx(ls)}]")
    sol.resultados = {"x̄": xbar, f"{nombre} crítico": crit, "Margen de error": E,
                      "Límite inferior": li, "Límite superior": ls}
    sol.conclusion = (f"Con un {ft(conf * 100, 2)} % de confianza, la media poblacional está entre "
                      f"{ft(li)} y {ft(ls)}.")
    sol.figuras = [plots.intervalo(xbar, li, ls, ee, "μ")]
    return sol


def resolver_prueba(e: Ejercicio) -> Solucion:
    alfa = e.alfa or (1 - e.confianza if e.confianza else 0.05)
    if alfa >= 1:
        alfa /= 100
    signo = {"bilateral": r"\neq", "izquierda": "<", "derecha": ">"}[e.cola]
    sol = Solucion(f"Prueba de hipótesis para la {e.parametro}")

    if e.parametro == "proporcion":
        p0 = float(_requerir(e.p, "proporción de la hipótesis nula p₀"))
        n = int(_requerir(e.n, "tamaño de muestra n"))
        ph = float(e.proporcion_muestral) if e.proporcion_muestral is not None else float(_requerir(e.a, "número de éxitos")) / n
        sol.paso("Plantear hipótesis", f"Nivel de significancia α = {ft(alfa)}.",
                 rf"H_0: p = {fx(p0)} \qquad H_1: p {signo} {fx(p0)}")
        ee = math.sqrt(p0 * (1 - p0) / n)
        est = (ph - p0) / ee
        sol.paso("Estadístico de prueba", "Bajo H₀ el estadístico sigue aproximadamente una normal estándar.",
                 rf"z = \frac{{\hat{{p}} - p_0}}{{\sqrt{{p_0(1-p_0)/n}}}} = \frac{{{fx(ph)} - {fx(p0)}}}{{\sqrt{{{fx(p0)}({fx(1 - p0)})/{n}}}}} = {fx(est)}")
        dist, gl, nombre = stats.norm(), None, "z"
        param_txt = f"p = {ft(p0)}"
    else:
        mu0 = float(_requerir(e.mu, "valor μ₀ de la hipótesis nula"))
        sol.paso("Plantear hipótesis", f"Nivel de significancia α = {ft(alfa)}.",
                 rf"H_0: \mu = {fx(mu0)} \qquad H_1: \mu {signo} {fx(mu0)}")
        xbar, s, n = _resumen_muestral(sol, e)
        if e.sigma is not None:
            sd, dist, gl, nombre, simb = float(e.sigma), stats.norm(), None, "z", r"\sigma"
        else:
            sd = _requerir(s, "desviación estándar (s o σ)")
            gl = n - 1
            dist, nombre, simb = stats.t(gl), "t", "s"
        est = (xbar - mu0) / (sd / math.sqrt(n))
        texto = ("σ conocida: se usa el estadístico Z." if gl is None
                 else f"σ desconocida: se usa el estadístico t con {gl} grados de libertad.")
        sol.paso("Estadístico de prueba", texto,
                 rf"{nombre} = \frac{{\bar{{x}} - \mu_0}}{{{simb}/\sqrt{{n}}}} = \frac{{{fx(xbar)} - {fx(mu0)}}}{{{fx(sd)}/\sqrt{{{n}}}}} = {fx(est)}")
        param_txt = f"μ = {ft(mu0)}"

    if e.cola == "bilateral":
        crit = dist.ppf(1 - alfa / 2)
        pval = 2 * (1 - dist.cdf(abs(est)))
        region = rf"|{nombre}| > {fx(crit)}"
        rechaza = abs(est) > crit
    elif e.cola == "izquierda":
        crit = dist.ppf(alfa)
        pval = dist.cdf(est)
        region = rf"{nombre} < {fx(crit)}"
        rechaza = est < crit
    else:
        crit = dist.ppf(1 - alfa)
        pval = 1 - dist.cdf(est)
        region = rf"{nombre} > {fx(crit)}"
        rechaza = est > crit
    sol.paso("Región crítica y valor p", f"Prueba {e.cola}.",
             rf"\text{{Se rechaza }} H_0 \text{{ si }} {region}",
             rf"\text{{valor p}} = {fx(pval, 6)}")
    decision = "Se **rechaza** H₀" if rechaza else "**No se rechaza** H₀"
    sol.paso("Decisión", f"{decision}: el estadístico {'cae' if rechaza else 'no cae'} en la región crítica "
             f"(valor p = {ft(pval, 4)} {'<' if rechaza else '≥'} α = {ft(alfa)}).")
    sol.resultados = {f"Estadístico {nombre}": float(est), "Valor crítico": float(crit),
                      "Valor p": float(pval), "Decisión": "Rechazar H₀" if rechaza else "No rechazar H₀"}
    sol.conclusion = (
        f"{'Hay' if rechaza else 'No hay'} evidencia estadística suficiente, al {ft(alfa * 100, 2)} % de "
        f"significancia, para afirmar que {param_txt.replace('=', {'bilateral': '≠', 'izquierda': '<', 'derecha': '>'}[e.cola])}."
    )
    sol.figuras = [plots.prueba(float(est), [float(crit)], e.cola, gl, nombre)]
    return sol


# --------------------------------------------------------------- expresiones

_PROHIBIDO = re.compile(r"__|\bimport\b|\blambda\b|\bexec\b|\beval\b|\bopen\b|\bos\b|\bsys\b|[;`]")
_LOCALES = {
    "C": sp.binomial, "nCr": sp.binomial, "comb": sp.binomial, "binom": sp.binomial,
    "nPr": sp.ff, "V": sp.ff, "factorial": sp.factorial, "sqrt": sp.sqrt, "raiz": sp.sqrt,
    "e": sp.E, "pi": sp.pi, "exp": sp.exp, "ln": sp.log, "log": sp.log, "Sum": sp.Sum,
    "sumatoria": sp.Sum, "x": sp.Symbol("x"), "k": sp.Symbol("k", integer=True), "n": sp.Symbol("n"),
}
_TRANSF = standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse(texto: str) -> sp.Expr:
    texto = texto.replace("×", "*").replace("÷", "/").replace("·", "*").replace("−", "-").replace("√", "sqrt")
    if _PROHIBIDO.search(texto):
        raise ValueError("La expresión contiene elementos no permitidos.")
    return parse_expr(texto, local_dict=dict(_LOCALES), transformations=_TRANSF)


def resolver_expresion(e: Ejercicio) -> Solucion:
    texto = (e.expresion or "").strip()
    if not texto:
        raise ValueError("No hay expresión para evaluar.")
    sol = Solucion("Evaluación de fórmula")
    try:
        if texto.count("=") == 1 and "==" not in texto:
            izq, der = (_parse(t) for t in texto.split("="))
            ecuacion = sp.Eq(izq, der)
            incog = sorted(ecuacion.free_symbols, key=str)
            sol.paso("Interpretar la ecuación", "Se reconoce la ecuación:", sp.latex(ecuacion))
            if not incog:
                sol.paso("Verificar", "", rf"\text{{La igualdad es }} \textbf{{{'verdadera' if sp.simplify(izq - der) == 0 else 'falsa'}}}")
                return sol
            var = incog[0]
            soluciones = sp.solve(ecuacion, var)
            sol.paso(f"Despejar {var}", "Se resuelve la ecuación con SymPy.",
                     *[rf"{sp.latex(var)} = {sp.latex(s)} \approx {fx(sp.N(s))}" if s.is_number else rf"{sp.latex(var)} = {sp.latex(s)}" for s in soluciones])
            sol.resultados = {f"{var}{i + 1 if len(soluciones) > 1 else ''}": (float(sp.N(s)) if s.is_real else str(s)) for i, s in enumerate(soluciones)}
            return sol
        expr = _parse(texto)
    except (SyntaxError, TypeError, sp.SympifyError, TokenError) as err:
        raise ValueError(f"No se pudo interpretar la expresión «{texto}»: {err}") from err

    sol.paso("Interpretar la fórmula", "La expresión reconocida es:", sp.latex(sp.sympify(expr, evaluate=False)))
    if expr.free_symbols:
        simp = sp.simplify(expr.doit())
        sol.paso("Simplificar", "La expresión tiene variables libres; se simplifica.", rf"= {sp.latex(simp)}")
        sol.resultados = {"Simplificada": str(simp)}
        return sol
    exacto = sp.nsimplify(expr.doit()) if expr.has(sp.Float) else expr.doit()
    valor = sp.N(exacto, 12)
    if exacto != expr:
        sol.paso("Evaluar", "Se calculan las operaciones (combinaciones, factoriales, potencias, etc.).",
                 rf"= {sp.latex(exacto)}")
    sol.paso("Resultado decimal", "", rf"\approx {fx(valor, 6)}")
    sol.resultados = {"Valor": float(valor) if valor.is_real else str(valor)}
    sol.conclusion = f"El valor de la expresión es {ft(valor, 6)}."
    return sol


RESOLVEDORES: dict[str, Callable[[Ejercicio], Solucion]] = {
    "descriptiva": resolver_descriptiva,
    "binomial": resolver_binomial,
    "poisson": resolver_poisson,
    "normal": resolver_normal,
    "regresion": resolver_regresion,
    "intervalo_confianza": resolver_intervalo,
    "prueba_hipotesis": resolver_prueba,
    "expresion": resolver_expresion,
}


def resolver(e: Ejercicio) -> Solucion:
    if e.tipo not in RESOLVEDORES:
        raise ValueError("No se identificó el tipo de ejercicio. Elige el tipo manualmente y completa los datos.")
    return RESOLVEDORES[e.tipo](e)
