# StatSolver: reconocimiento de ejercicios estadísticos

StatSolver lee la foto de un ejercicio de estadística, que puede estar escrito a mano, y entrega la solución explicada paso a paso, con fórmulas en LaTeX, tablas y gráficos.

**Tecnologías:** Python, Streamlit, SymPy, SciPy, Matplotlib y la visión de Claude para reconocer el texto matemático.

## Cómo funciona

```
 Foto ──► Reconocimiento (Claude Vision) ──┐
                                            ├─► Ejercicio (JSON) ──► Revisión del usuario ──► Solucionador (SymPy/SciPy) ──► Pasos + gráficos
 Texto ──► Intérprete local (regex) ───────┘
```

1. **Reconocimiento.** El modelo de visión transcribe el enunciado y extrae el tipo de ejercicio y sus datos con una salida estructurada (el modelo `Ejercicio` de [statsolver/models.py](statsolver/models.py)). Solo lee el ejercicio; no lo resuelve.
2. **Revisión.** La app muestra lo reconocido en un formulario editable para que el estudiante corrija un dígito mal leído antes de resolver.
3. **Solución.** [statsolver/solvers.py](statsolver/solvers.py) hace los cálculos de forma determinista. SymPy se encarga de la aritmética exacta, por ejemplo con fracciones y combinaciones, y SciPy de los cuantiles y las funciones de distribución. Así los resultados son verificables y no dependen de la IA.

## Tipos de ejercicio

| Tipo | Qué calcula | Gráfico |
|---|---|---|
| Estadística descriptiva | Tabla de frecuencias (con Sturges), media, mediana, moda, rango, varianza, desviación, CV, cuartiles y atípicos | Histograma y diagrama de caja |
| Binomial | P(X = k), P(X ≤ k), P(X ≥ k), entre, percentil (usa el complemento cuando conviene), media y varianza | Barras con el área pedida resaltada |
| Poisson | Igual que la binomial, con λ | Barras |
| Normal | Estandarización a Z, Φ(z) (también con tabla a 2 decimales), percentiles, media muestral (TLC) | Curva con el área sombreada |
| Regresión lineal | Tabla de sumas, b₁, b₀, recta, r, r² y predicción | Dispersión y recta |
| Intervalo de confianza | Media (Z o t según se conozca σ) y proporción | Intervalo sobre la distribución |
| Prueba de hipótesis | Media (Z/t) y proporción, bilateral o de una cola, valor crítico, valor p y decisión | Región de rechazo |
| Fórmula / expresión | Combinaciones `C(n,k)`, `nPr`, factoriales `n!`, potencias, sumatorias y ecuaciones con `=` | — |

## Instalación y uso

Requisitos: Python 3.10 o superior.

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

En Windows también basta con hacer doble clic en `iniciar.bat`.

Para reconocer fotos necesitas una API key de Anthropic. Hay dos formas de darla:

- copiar `.env.example` como `.env` y poner la clave ahí, o
- escribirla en la barra lateral de la app.

La pestaña **Escribir el ejercicio** funciona sin clave. Acepta entradas como estas:

```
datos: 12, 15, 11, 18, 15, 20
binomial n=10 p=0.3 P(X<=2)
B(20, 0.5) al menos 12
poisson λ=4 a lo más 3
normal μ=70 σ=8 P(65 < X < 80)
x: 1 2 3 4 5   y: 2.1 3.9 6.2 7.8 10.1   predecir x=6
intervalo de confianza media x̄=52 s=6 n=36 confianza 95%
hipotesis H0: μ=50 H1: μ>50 x̄=52 s=6 n=36 α=0.05
C(10,3)*0.3^3*0.7^7
```

Si usas coma decimal, separa los datos con punto y coma: `2,5; 3,1; 4,0`.

## Usarla desde el celular

1. Conecta el celular al **mismo Wi-Fi** que la PC y abre la app en la PC.
2. En la barra lateral, en **📱 Abrir en el celular**, escanea el código QR (o escribe la dirección `http://IP-de-la-PC:8501` que aparece ahí y en la consola como *Network URL*).
3. Para la foto, toca **Subir o tomar foto** y elige **Cámara**: se usa la cámara trasera con su resolución completa.

Notas:

- La opción **Cámara en vivo** necesita HTTPS, así que en el celular (que entra por `http://`) solo funciona *Subir o tomar foto*.
- Si el celular no carga la página, revisa que la red de Windows esté como **Privada** y que el Firewall permita Python en redes privadas.
- Cualquiera en tu Wi-Fi podría abrir la app y usar tu API key. Para evitarlo, define `STATSOLVER_PASSWORD` en el `.env` y la app pedirá esa contraseña al entrar.

## Publicarla en internet (disponible siempre)

Para usarla desde cualquier lugar y a cualquier hora, aunque la PC esté apagada, se publica gratis en **Streamlit Community Cloud**:

1. Sube esta carpeta a un repositorio de GitHub. Puede ser privado. El `.gitignore` deja fuera `.env` y `.venv`.
2. Entra a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub y pulsa **Create app**.
3. Elige el repositorio, la rama `main` y el archivo `app.py`.
4. En **Advanced settings**, elige Python 3.12 y en **Secrets** pega (con tus valores):

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   STATSOLVER_PASSWORD = "elige-una-clave"
   ```

5. Pulsa **Deploy**. Recibirás un link `https://<nombre>.streamlit.app` que puedes guardar en la pantalla de inicio del celular.

Por seguridad, la app publicada solo usa la API key del servidor si `STATSOLVER_PASSWORD` está definida. Sin contraseña, cada visitante tendría que escribir su propia clave. Si la app no recibe visitas durante varios días, Streamlit la pone en reposo; al abrirla aparece un botón para despertarla, que tarda menos de un minuto.

## Pruebas

```bash
pytest -q
```

Las pruebas comparan cada solucionador con los valores de referencia de SciPy y NumPy y también revisan el intérprete de texto.

## Estructura

```
StatSolver/
├── app.py                  Interfaz Streamlit
├── statsolver/
│   ├── models.py           Esquema del ejercicio y de la solución
│   ├── recognizer.py       Foto → Ejercicio (visión de Claude, salida estructurada)
│   ├── parser.py           Texto → Ejercicio (sin API)
│   ├── solvers.py          Soluciones paso a paso (SymPy / SciPy)
│   ├── plots.py            Gráficos (Matplotlib)
│   └── reporte.py          Exportación a Markdown
└── tests/test_solvers.py
```

## Notas

- El reconocimiento usa el modelo `claude-opus-5-5` con *fallbacks* del lado del servidor: si el modelo declina una imagen, la API la reintenta automáticamente con otro modelo.
- Antes de enviarla, la foto se reduce a un máximo de 1568 px por lado y se corrige su orientación EXIF.
- Cada foto reconocida hace una llamada a la API, con su costo correspondiente. La entrada de texto y todos los cálculos son locales y gratuitos.
