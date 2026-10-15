@echo off
REM Inicia StatSolver en el navegador. La primera vez crea el entorno e instala dependencias.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    py -3 -m venv .venv
)
.venv\Scripts\python.exe -m pip install -q -r requirements.txt
echo.
echo Para abrirla en el celular: escanea el codigo QR de la barra lateral
echo o entra a la "Network URL" que aparece abajo (mismo Wi-Fi que esta PC).
echo.
.venv\Scripts\python.exe -m streamlit run app.py
