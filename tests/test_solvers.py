"""Pruebas de los solucionadores contra valores conocidos (SciPy / NumPy)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from statsolver import interpretar_texto, resolver  # noqa: E402
from statsolver.models import Ejercicio  # noqa: E402
from statsolver.reporte import a_markdown  # noqa: E402


def _prob(sol):
    return next(v for k, v in sol.resultados.items() if k.startswith("P(") or k == "Probabilidad")


def test_descriptiva():
    datos = [12, 15, 11, 18, 15, 20, 14, 15, 17, 13]
    sol = resolver(Ejercicio.vacio(tipo="descriptiva", datos=datos))
    r = sol.resultados
    assert r["Media"] == pytest.approx(np.mean(datos))
    assert r["Mediana"] == pytest.approx(np.median(datos))
    assert r["Varianza"] == pytest.approx(np.var(datos, ddof=1))
    assert r["Moda"] == "15"
    assert len(sol.figuras) == 2


def test_descriptiva_poblacion_y_agrupada():
    datos = list(np.random.default_rng(1).normal(50, 10, 60).round(1))
    sol = resolver(Ejercicio.vacio(tipo="descriptiva", datos=datos, es_poblacion=True))
    assert sol.resultados["Varianza"] == pytest.approx(np.var(datos), rel=1e-9)
    assert "Clase" in sol.pasos[1].tabla.columns


@pytest.mark.parametrize("consulta,a,b,esperado", [
    ("igual", 3, None, stats.binom.pmf(3, 10, 0.3)),
    ("menor_igual", 2, None, stats.binom.cdf(2, 10, 0.3)),
    ("menor", 2, None, stats.binom.cdf(1, 10, 0.3)),
    ("mayor_igual", 2, None, stats.binom.sf(1, 10, 0.3)),
    ("mayor", 7, None, stats.binom.sf(7, 10, 0.3)),
    ("entre", 2, 5, stats.binom.cdf(5, 10, 0.3) - stats.binom.cdf(1, 10, 0.3)),
])
def test_binomial(consulta, a, b, esperado):
    sol = resolver(Ejercicio.vacio(tipo="binomial", n=10, p=0.3, consulta=consulta, a=a, b=b))
    assert _prob(sol) == pytest.approx(esperado)


def test_poisson():
    sol = resolver(Ejercicio.vacio(tipo="poisson", lam=4, consulta="mayor_igual", a=2))
    assert _prob(sol) == pytest.approx(stats.poisson.sf(1, 4))


def test_normal():
    sol = resolver(Ejercicio.vacio(tipo="normal", mu=70, sigma=8, consulta="entre", a=65, b=80))
    assert _prob(sol) == pytest.approx(stats.norm.cdf(80, 70, 8) - stats.norm.cdf(65, 70, 8))
    sol = resolver(Ejercicio.vacio(tipo="normal", mu=70, sigma=8, n=16, consulta="mayor", a=73))
    assert _prob(sol) == pytest.approx(stats.norm.sf(73, 70, 2))
    sol = resolver(Ejercicio.vacio(tipo="normal", mu=70, sigma=8, consulta="percentil", probabilidad=0.9))
    assert sol.resultados["x"] == pytest.approx(stats.norm.ppf(0.9, 70, 8))


def test_regresion():
    x, y = [1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1]
    sol = resolver(Ejercicio.vacio(tipo="regresion", datos=x, datos_y=y, x_prediccion=6))
    res = stats.linregress(x, y)
    assert sol.resultados["b₁ (pendiente)"] == pytest.approx(res.slope)
    assert sol.resultados["b₀ (intercepto)"] == pytest.approx(res.intercept)
    assert sol.resultados["r"] == pytest.approx(res.rvalue)
    assert sol.resultados["ŷ predicho"] == pytest.approx(res.intercept + 6 * res.slope)


def test_intervalo_t_y_proporcion():
    sol = resolver(Ejercicio.vacio(tipo="intervalo_confianza", media_muestral=52, desviacion_muestral=6,
                                   n=36, confianza=0.95))
    E = stats.t.ppf(0.975, 35) * 6 / 6
    assert sol.resultados["Límite inferior"] == pytest.approx(52 - E)
    sol = resolver(Ejercicio.vacio(tipo="intervalo_confianza", parametro="proporcion", n=200, a=120,
                                   confianza=0.9))
    E = stats.norm.ppf(0.95) * math.sqrt(0.6 * 0.4 / 200)
    assert sol.resultados["Límite superior"] == pytest.approx(0.6 + E)


def test_prueba_hipotesis():
    sol = resolver(Ejercicio.vacio(tipo="prueba_hipotesis", mu=50, media_muestral=52, desviacion_muestral=6,
                                   n=36, alfa=0.05, cola="derecha"))
    t = (52 - 50) / (6 / 6)
    assert sol.resultados["Valor p"] == pytest.approx(stats.t.sf(t, 35))
    assert sol.resultados["Decisión"] == "Rechazar H₀"


def test_expresion_y_ecuacion():
    sol = resolver(Ejercicio.vacio(tipo="expresion", expresion="C(10,3)*0.3^3*0.7^7"))
    assert sol.resultados["Valor"] == pytest.approx(stats.binom.pmf(3, 10, 0.3))
    sol = resolver(Ejercicio.vacio(tipo="expresion", expresion="5! / (2! * 3!)"))
    assert sol.resultados["Valor"] == pytest.approx(10)
    sol = resolver(Ejercicio.vacio(tipo="expresion", expresion="x/5 + 1 = 3"))
    assert sol.resultados["x"] == pytest.approx(10)
    with pytest.raises(ValueError):
        resolver(Ejercicio.vacio(tipo="expresion", expresion="__import__('os')"))


@pytest.mark.parametrize("texto,tipo,campos", [
    ("binomial n=10 p=0.3 P(X<=2)", "binomial", dict(n=10, p=0.3, consulta="menor_igual", a=2)),
    ("B(20, 0.5) al menos 12", "binomial", dict(n=20, p=0.5, consulta="mayor_igual", a=12)),
    ("poisson λ=4 a lo más 3", "poisson", dict(lam=4, consulta="menor_igual", a=3)),
    ("normal μ=70 σ=8 P(65 < X < 80)", "normal", dict(mu=70, sigma=8, consulta="entre", a=65, b=80)),
    ("datos: 4, 8, 15, 16, 23, 42", "descriptiva", dict(datos=[4, 8, 15, 16, 23, 42])),
    ("2,5; 3,1; 4,0", "descriptiva", dict(datos=[2.5, 3.1, 4.0])),
    ("x: 1 2 3 4\ny: 2 4 5 8\npredecir x=6", "regresion", dict(datos=[1, 2, 3, 4], datos_y=[2, 4, 5, 8], x_prediccion=6)),
    ("intervalo de confianza media x̄=52 s=6 n=36 confianza 95%", "intervalo_confianza",
     dict(media_muestral=52, desviacion_muestral=6, n=36, confianza=0.95)),
    ("hipotesis H0: μ=50 H1: μ>50 x̄=52 s=6 n=36 α=0.05", "prueba_hipotesis",
     dict(mu=50, cola="derecha", media_muestral=52, n=36)),
    ("C(10,3)*0.3^3", "expresion", dict(expresion="C(10,3)*0.3^3")),
])
def test_parser(texto, tipo, campos):
    e = interpretar_texto(texto)
    assert e.tipo == tipo
    for k, v in campos.items():
        assert getattr(e, k) == (pytest.approx(v) if isinstance(v, (float, int, list)) else v), k


def test_reporte_markdown():
    e = interpretar_texto("datos: 1, 2, 3, 4, 5")
    md = a_markdown(e, resolver(e))
    assert "## Solución paso a paso" in md and "$$" in md
