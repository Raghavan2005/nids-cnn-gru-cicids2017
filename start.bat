@echo off
REM Start the NIDS Command Center (PySide6 app + phone web server).
REM Creates a virtualenv and installs dependencies on first run.
cd /d "%~dp0T014_Project"

set VENV=..\.venv

if not exist "%VENV%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv "%VENV%" || (echo Python 3 not found on PATH & pause & exit /b 1)
)
call "%VENV%\Scripts\activate.bat"

python -c "import PySide6, segno, tensorflow, numpy, pandas" 2>nul
if errorlevel 1 (
    echo Installing dependencies ^(first run only^)...
    pip install -r requirements.txt
    pip install tensorflow keras PySide6 segno
)

REM Optional: set NIDS_PORT=9000 to change the port (default 8000)
python webapp\app.py %*
if errorlevel 1 pause
