@echo off
REM Inicia StatSolver y la publica con un link HTTPS temporal (Cloudflare Tunnel)
REM para abrirla desde el celular con Wi-Fi o datos moviles. Cierra esta ventana para apagar el link.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creando entorno virtual...
    py -3 -m venv .venv
)
.venv\Scripts\python.exe -m pip install -q -r requirements.txt

REM El link es publico: cualquiera que lo tenga podria entrar, asi que se exige contrasena.
if "%STATSOLVER_PASSWORD%"=="" (
    findstr /b /c:"STATSOLVER_PASSWORD=" .env >nul 2>&1 || set /p STATSOLVER_PASSWORD=Elige una contrasena para entrar a la app:
)

if not exist "herramientas\cloudflared.exe" (
    echo Descargando cloudflared desde el repositorio oficial de Cloudflare...
    if not exist herramientas mkdir herramientas
    curl -L -o herramientas\cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe || (
        echo No se pudo descargar cloudflared. & pause & exit /b 1
    )
)

start "StatSolver" /min .venv\Scripts\python.exe -m streamlit run app.py --server.headless true

echo.
echo ================================================================
echo  En unos segundos aparecera un link https://....trycloudflare.com
echo  Abrelo en el celular (funciona con Wi-Fi o con datos moviles).
echo  Cierra esta ventana para apagar el link.
echo ================================================================
echo.
herramientas\cloudflared.exe tunnel --no-autoupdate --url http://localhost:8501
