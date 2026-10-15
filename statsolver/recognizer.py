"""Reconocimiento de la foto del ejercicio con el modelo de visión de Claude.

El modelo solo transcribe y estructura el enunciado (tipo de ejercicio + datos);
los cálculos los hace `solvers.py` de forma determinista.
"""
from __future__ import annotations

import base64
import io
import os

import anthropic
from PIL import Image, ImageOps

from .models import Ejercicio

MODELO = "claude-opus-5-5"
LADO_MAX = 1568  # px; imágenes más grandes no aportan detalle extra al modelo

INSTRUCCIONES = """Eres el módulo de reconocimiento de StatSolver. Recibes la foto de un ejercicio de \
estadística (a menudo manuscrito, en español) y debes transcribirlo y extraer sus datos en el esquema \
JSON indicado. No resuelvas el ejercicio: otro módulo hace los cálculos a partir de lo que extraigas, \
así que la fidelidad de los números es lo más importante.

Cómo llenar los campos:
- transcripcion: el enunciado completo tal como se lee. latex: la fórmula principal en LaTeX ("" si no hay).
- tipo: descriptiva (lista de datos para media, mediana, varianza...), binomial, poisson, normal, \
regresion (pares x, y), intervalo_confianza, prueba_hipotesis, expresion (una fórmula o cálculo suelto, \
p. ej. combinaciones, factoriales, sumatorias o una ecuación), desconocido.
- consulta + a + b: la probabilidad pedida. "P(X=3)" → igual, a=3. "a lo más 3" / "como máximo 3" → \
menor_igual. "menos de 3" → menor. "al menos 3" → mayor_igual. "más de 3" → mayor. "entre 2 y 5" → \
entre, a=2, b=5. Si se pide el valor x para una probabilidad dada → percentil y probabilidad. Si no se \
pide probabilidad → ninguna.
- normal: mu y sigma de la población; si se pregunta por la media de una muestra de tamaño n, pon n.
- intervalo_confianza / prueba_hipotesis: parametro (media o proporcion), media_muestral, \
desviacion_muestral (s), sigma solo si la desviación poblacional es conocida, n, confianza o alfa \
como decimal (95 % → 0.95). Para pruebas, mu es μ₀ (o p es p₀ en proporciones) y cola según H₁ \
(≠ bilateral, < izquierda, > derecha). Si dan el número de éxitos de una proporción, ponlo en a.
- regresion: datos = valores de x, datos_y = valores de y, x_prediccion si se pide predecir.
- expresion: la fórmula en sintaxis de Python/SymPy, usando C(n,k) para combinaciones, nPr(n,k) para \
permutaciones, factorial(n) o n!, sqrt(), ** para potencias, Sum(f, (k, a, b)) para sumatorias. \
Una ecuación se escribe con un solo "=".
- Usa punto decimal. Convierte porcentajes a decimales cuando sean probabilidades o proporciones.
- Los campos que no aparezcan van en null (o lista vacía / cadena vacía). Si un número es ilegible, \
no lo inventes: déjalo en null y menciónalo en la transcripción con [ilegible]."""


class ErrorReconocimiento(RuntimeError):
    pass


def hay_credenciales() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def preparar_imagen(contenido: bytes) -> tuple[str, str]:
    """Corrige la orientación EXIF, reduce el tamaño y codifica en base64 (JPEG)."""
    img = Image.open(io.BytesIO(contenido))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((LADO_MAX, LADO_MAX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def reconocer_imagen(contenido: bytes, api_key: str | None = None) -> Ejercicio:
    datos, media_type = preparar_imagen(contenido)
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    try:
        respuesta = client.beta.messages.parse(
            model=MODELO,
            max_tokens=16000,
            system=INSTRUCCIONES,
            output_config={"effort": "medium"},
            # Si el modelo principal declina la solicitud, la API la reintenta con otro modelo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_format=Ejercicio,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": datos}},
                    {"type": "text", "text": "Transcribe y estructura este ejercicio de estadística."},
                ],
            }],
        )
    except anthropic.AuthenticationError as err:
        raise ErrorReconocimiento("La API key de Anthropic no es válida.") from err
    except anthropic.RateLimitError as err:
        raise ErrorReconocimiento("Se alcanzó el límite de uso de la API. Intenta en unos segundos.") from err
    except anthropic.APIStatusError as err:
        raise ErrorReconocimiento(f"Error de la API ({err.status_code}): {err.message}") from err
    except anthropic.APIConnectionError as err:
        raise ErrorReconocimiento("No se pudo conectar con la API de Anthropic. Revisa tu conexión.") from err

    if respuesta.stop_reason == "refusal":
        raise ErrorReconocimiento("El modelo no pudo procesar esta imagen.")
    if respuesta.stop_reason == "max_tokens" or respuesta.parsed_output is None:
        raise ErrorReconocimiento("No se obtuvo una respuesta completa del reconocimiento. Intenta de nuevo.")
    return respuesta.parsed_output
