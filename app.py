"""StatSolver — interfaz web (Streamlit).

Ejecutar con:  streamlit run app.py
"""
from __future__ import annotations

import hmac
import io
import os
import socket

import matplotlib.pyplot as plt
import segno
import streamlit as st
from dotenv import load_dotenv

from statsolver import interpretar_texto, resolver
from statsolver.models import CONSULTAS, TIPOS, Ejercicio
from statsolver.recognizer import ErrorReconocimiento, reconocer_imagen
from statsolver.reporte import a_markdown
from statsolver.solvers import ft

load_dotenv()
st.set_page_config(page_title="StatSolver", page_icon="📊", layout="centered")

ESTILOS = """
<style>
.block-container {padding-top: 2.5rem; padding-bottom: 4rem; max-width: 900px;}
.hero {padding: 1.4rem 1.5rem; border-radius: 1rem; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, rgba(37,99,235,.12), rgba(124,58,237,.10));
       border: 1px solid rgba(37,99,235,.18);}
.hero .marca {font-weight: 700; font-size: .85rem; letter-spacing: .04em; text-transform: uppercase;
              color: #2563eb;}
.hero .titulo {font-size: 2rem; font-weight: 800; line-height: 1.15; margin: .35rem 0 .5rem;}
.hero p {margin: 0 0 .9rem; opacity: .8;}
.hero .pasos {display: flex; flex-wrap: wrap; gap: .5rem;}
.hero .pasos span {font-size: .85rem; padding: .3rem .7rem; border-radius: 999px;
                   background: rgba(37,99,235,.10); border: 1px solid rgba(37,99,235,.20);}
@media (max-width: 640px) {
  .block-container {padding: 1rem .9rem 4rem;}
  .hero {padding: 1.1rem 1rem;}
  .hero .titulo {font-size: 1.45rem;}
  .stTabs [data-baseweb="tab"] {font-size: .95rem;}
}
</style>
"""

HERO = """
<div class="hero">
  <div class="marca">📊 StatSolver</div>
  <div class="titulo">Resuelve ejercicios de estadística con una foto</div>
  <p>Fotografía el enunciado (impreso o a mano), revisa los datos y obtén la solución paso a paso.</p>
  <div class="pasos"><span>1 · Foto o texto</span><span>2 · Revisa los datos</span><span>3 · Solución</span></div>
</div>
"""

EJEMPLOS = {
    "Descriptiva": "datos: 12, 15, 11, 18, 15, 20, 14, 15, 17, 13",
    "Binomial": "binomial n=10 p=0.3 P(X<=2)",
    "Poisson": "poisson lambda=4 al menos 2",
    "Normal": "normal media=70 sigma=8 P(65 < X < 80)",
    "Regresión": "x: 1, 2, 3, 4, 5\ny: 2.1, 3.9, 6.2, 7.8, 10.1\npredecir x=6",
    "Intervalo": "intervalo de confianza media xbar=52 s=6 n=36 confianza 95%",
    "Hipótesis": "prueba de hipotesis H0: mu=50 H1: mu>50 xbar=52 s=6 n=36 alfa=0.05",
    "Fórmula": "C(10,3)*0.3^3*0.7^7",
}

# Campos editables por tipo de ejercicio.
CAMPOS = {
    "descriptiva": ["datos", "es_poblacion"],
    "binomial": ["n", "p", "consulta", "a", "b", "probabilidad"],
    "poisson": ["lam", "consulta", "a", "b", "probabilidad"],
    "normal": ["mu", "sigma", "n", "consulta", "a", "b", "probabilidad"],
    "regresion": ["datos", "datos_y", "x_prediccion"],
    "intervalo_confianza": ["parametro", "datos", "media_muestral", "desviacion_muestral", "sigma", "n",
                            "proporcion_muestral", "a", "confianza"],
    "prueba_hipotesis": ["parametro", "cola", "mu", "p", "datos", "media_muestral", "desviacion_muestral",
                         "sigma", "n", "proporcion_muestral", "a", "alfa"],
    "expresion": ["expresion"],
    "desconocido": [],
}

ETIQUETAS = {
    "datos": "Datos (separados por coma)", "datos_y": "Valores de Y", "es_poblacion": "Los datos son una población",
    "n": "n (ensayos o tamaño de muestra)", "p": "p (probabilidad de éxito / p₀)", "lam": "λ (promedio)",
    "mu": "μ (media poblacional / μ₀)", "sigma": "σ (desv. poblacional, si se conoce)",
    "media_muestral": "x̄ (media muestral)", "desviacion_muestral": "s (desv. muestral)",
    "proporcion_muestral": "p̂ (proporción muestral)", "a": "a (valor de la consulta / nº de éxitos)",
    "b": "b (límite superior)", "consulta": "Probabilidad pedida", "probabilidad": "Probabilidad acumulada (percentil)",
    "confianza": "Nivel de confianza (0.95)", "alfa": "α (significancia)", "cola": "Hipótesis alternativa",
    "parametro": "Parámetro", "x_prediccion": "x para predecir ŷ", "expresion": "Expresión (sintaxis SymPy)",
}
LISTAS = {"datos", "datos_y"}
OPCIONES = {
    "consulta": list(CONSULTAS),
    "cola": ["bilateral", "izquierda", "derecha"],
    "parametro": ["media", "proporcion"],
}
NOMBRES_OPCION = {**CONSULTAS, "bilateral": "≠ (bilateral)", "izquierda": "< (cola izquierda)",
                  "derecha": "> (cola derecha)", "media": "Media μ", "proporcion": "Proporción p"}


def _num(texto: str):
    texto = texto.strip().replace(",", ".")
    return float(texto) if texto else None


def _lista(texto: str) -> list[float]:
    partes = texto.replace(";", ",").replace("\n", ",").split(",")
    return [float(p.strip()) for p in partes if p.strip()]


def _api_key() -> str:
    """Clave de la sesión (escrita en la app) o, si no hay, la del .env / secretos del servidor.

    La clave del servidor solo se usa desde esta PC o si la app pide contraseña: así, al publicarla,
    nadie que encuentre el link puede gastar el saldo de la cuenta.
    """
    if st.session_state.get("api_key"):
        return st.session_state.api_key
    if _es_local() or os.environ.get("STATSOLVER_PASSWORD"):
        return os.environ.get("ANTHROPIC_API_KEY", "")
    return ""


def _ip_local() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no envía nada; solo elige la interfaz de la red local
            return s.getsockname()[0]
    except OSError:
        return None


def _es_local() -> bool:
    host = (st.context.headers.get("Host") or "").split(":")[0]
    return host in ("localhost", "127.0.0.1", "[::1]")


def _acceso() -> None:
    """Pide la contraseña si STATSOLVER_PASSWORD está definida (útil al abrir la app en la red)."""
    clave = os.environ.get("STATSOLVER_PASSWORD")
    if not clave or st.session_state.get("autorizado"):
        return
    st.markdown(HERO, unsafe_allow_html=True)
    with st.form("acceso"):
        intento = st.text_input("Contraseña", type="password")
        entrar = st.form_submit_button("Entrar", type="primary", width="stretch")
    if entrar:
        if hmac.compare_digest(intento.encode(), clave.encode()):
            st.session_state.autorizado = True
            st.rerun()
        st.error("Contraseña incorrecta.")
    st.stop()


def panel_celular() -> None:
    """QR con la dirección de la red local, para abrir la app desde el celular."""
    ip = _ip_local()
    if ip is None:
        st.caption("No se detectó una red local.")
        return
    url = f"http://{ip}:{st.get_option('server.port')}"
    buf = io.BytesIO()
    segno.make(url, error="m").save(buf, kind="png", scale=6, border=2)
    st.image(buf.getvalue(), width=180)
    st.code(url, language=None)
    st.caption("Escanea el código con el celular conectado al mismo Wi-Fi que esta PC.")


def editor(e: Ejercicio) -> Ejercicio | None:
    """Formulario para revisar y corregir lo reconocido antes de resolver."""
    with st.form("editor"):
        tipo = st.selectbox("Tipo de ejercicio", list(TIPOS), index=list(TIPOS).index(e.tipo),
                            format_func=TIPOS.get)
        st.caption("Si cambias el tipo, pulsa «Aplicar cambios» para ver sus campos.")
        valores = e.model_dump()
        campos = CAMPOS[e.tipo]
        columnas = st.columns(2)
        nuevos = {}
        for i, campo in enumerate(campos):
            col = columnas[i % 2] if campo not in LISTAS | {"expresion"} else st
            actual = valores[campo]
            if campo in LISTAS:
                nuevos[campo] = col.text_input(ETIQUETAS[campo], ", ".join(ft(v, 6) for v in actual))
            elif campo == "es_poblacion":
                nuevos[campo] = col.checkbox(ETIQUETAS[campo], value=actual)
            elif campo in OPCIONES:
                ops = OPCIONES[campo]
                nuevos[campo] = col.selectbox(ETIQUETAS[campo], ops, index=ops.index(actual),
                                              format_func=lambda o: NOMBRES_OPCION.get(o, o))
            elif campo == "expresion":
                nuevos[campo] = col.text_input(ETIQUETAS[campo], actual)
            else:
                nuevos[campo] = col.text_input(ETIQUETAS[campo], "" if actual is None else ft(actual, 6))
        c1, c2 = st.columns([1, 1])
        aplicar = c1.form_submit_button("Aplicar cambios", width="stretch")
        resolver_btn = c2.form_submit_button("Resolver paso a paso", type="primary", width="stretch")

    if not (aplicar or resolver_btn):
        return None
    try:
        for campo, valor in nuevos.items():
            if campo in LISTAS:
                valores[campo] = _lista(valor)
            elif campo in ("es_poblacion", "expresion") or campo in OPCIONES:
                valores[campo] = valor
            else:
                valores[campo] = _num(valor)
        valores["tipo"] = tipo
        nuevo = Ejercicio(**valores)
    except ValueError as err:
        st.error(f"Revisa los datos: {err}")
        return None
    st.session_state.ejercicio = nuevo
    st.session_state.resolver = resolver_btn
    if aplicar:
        st.rerun()
    return nuevo


def mostrar_solucion(e: Ejercicio) -> None:
    try:
        sol = resolver(e)
    except (ValueError, ZeroDivisionError) as err:
        st.error(f"No se pudo resolver: {err}")
        return

    st.header(sol.titulo, anchor="solucion")
    numericos = [(k, v) for k, v in sol.resultados.items()]
    if numericos:
        cols = st.columns(min(3, len(numericos)))
        for i, (k, v) in enumerate(numericos):
            with cols[i % len(cols)].container(border=True):
                st.metric(k, ft(v, 4) if isinstance(v, (int, float)) else str(v))

    if sol.conclusion:
        st.success(f"**Conclusión:** {sol.conclusion}")
    for adv in sol.advertencias:
        st.warning(adv)

    st.subheader("Solución paso a paso")
    for i, paso in enumerate(sol.pasos, 1):
        with st.container(border=True):
            st.markdown(f"**Paso {i}. {paso.titulo}**")
            if paso.texto:
                st.markdown(paso.texto)
            for ltx in paso.latex:
                st.latex(ltx)
            if paso.tabla is not None:
                st.dataframe(paso.tabla, hide_index=True, width="stretch")

    if sol.figuras:
        st.subheader("Gráficos")
        for fig in sol.figuras:
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    st.download_button("⬇️ Descargar solución (Markdown)", a_markdown(e, sol), file_name="solucion_statsolver.md",
                       mime="text/markdown", width="stretch")


# ------------------------------------------------------------------ interfaz

st.markdown(ESTILOS, unsafe_allow_html=True)
_acceso()

with st.sidebar:
    st.title("📊 StatSolver")
    st.caption("Fotografía un ejercicio de estadística y obtén la solución explicada paso a paso.")
    if _api_key():
        st.success("API key configurada", icon="🔑")
    else:
        st.info("Sin API key: la foto no se puede reconocer, pero la pestaña de texto funciona.", icon="🔑")
    if _es_local():
        with st.expander("📱 Abrir en el celular", expanded=True):
            panel_celular()
    st.markdown(
        "**Tipos soportados**\n\n" + "\n".join(f"- {v}" for k, v in TIPOS.items() if k != "desconocido")
    )
    st.caption("El reconocimiento de la foto usa la visión de Claude; todos los cálculos se hacen "
               "localmente con SymPy y SciPy.")

st.markdown(HERO, unsafe_allow_html=True)

tab_foto, tab_texto = st.tabs(["📷 Foto", "⌨️ Escribir"])

with tab_foto:
    if not _api_key():
        with st.expander("🔑 Configura tu API key de Anthropic para reconocer fotos", expanded=True):
            # El campo se oculta en cuanto hay clave, así que se copia a un estado que no es de widget.
            st.text_input("API key", type="password", key="api_key_campo", placeholder="sk-ant-...",
                          on_change=lambda: st.session_state.update(api_key=st.session_state.api_key_campo.strip()),
                          help="Se guarda solo en esta sesión. También puedes ponerla en un archivo .env.")
            st.caption("Sin clave, usa la pestaña ⌨️ Escribir.")

    fuente = st.segmented_control("Origen de la imagen", ["📁 Subir o tomar foto", "📸 Cámara en vivo"],
                                  default="📁 Subir o tomar foto", label_visibility="collapsed")
    if fuente == "📸 Cámara en vivo":
        st.caption("La cámara en vivo solo funciona en esta PC o con HTTPS. En el celular usa "
                   "«Subir o tomar foto»: te ofrecerá abrir la cámara trasera.")
        archivo = st.camera_input("Toma una foto del ejercicio")
    else:
        archivo = st.file_uploader("Foto del ejercicio", type=["jpg", "jpeg", "png", "webp"],
                                   help="En el celular, al tocar el botón puedes elegir la cámara.")
    if archivo is not None:
        st.image(archivo, width="stretch")
        if st.button("🔍 Reconocer ejercicio", type="primary", disabled=not _api_key(), width="stretch"):
            with st.spinner("Leyendo el ejercicio…"):
                try:
                    st.session_state.ejercicio = reconocer_imagen(archivo.getvalue(), api_key=_api_key())
                    st.session_state.resolver = False
                except ErrorReconocimiento as err:
                    st.error(str(err))


def _cargar_ejemplo() -> None:
    if st.session_state.ejemplo:
        st.session_state.texto = EJEMPLOS[st.session_state.ejemplo]


with tab_texto:
    st.pills("Ejemplos", list(EJEMPLOS), key="ejemplo", on_change=_cargar_ejemplo)
    texto = st.text_area("Enunciado", key="texto", height=120,
                         placeholder="Ej.: binomial n=10 p=0.3 P(X<=2)")
    if st.button("✨ Interpretar", type="primary", width="stretch") and texto.strip():
        st.session_state.ejercicio = interpretar_texto(texto)
        st.session_state.resolver = False

ejercicio: Ejercicio | None = st.session_state.get("ejercicio")
if ejercicio is not None:
    st.divider()
    st.subheader("Ejercicio reconocido")
    with st.container(border=True):
        if ejercicio.transcripcion:
            st.markdown(f"> {ejercicio.transcripcion}".replace("\n", "\n> "))
        if ejercicio.latex:
            st.latex(ejercicio.latex)
        if ejercicio.pregunta:
            st.caption(f"Se pide: {ejercicio.pregunta}")
    st.markdown("Revisa los datos reconocidos y corrígelos si hace falta:")
    editor(ejercicio)
    if st.session_state.get("resolver"):
        st.divider()
        mostrar_solucion(st.session_state.ejercicio)
