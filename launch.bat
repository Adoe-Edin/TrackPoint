@echo off
:: =============================================================================
:: PrepByEKS -- TAPIR Point Tracker v2  (DEV COPY)
:: LAUNCHER  --  Double-click this to start the app
::
:: This is the development folder. It uses the system-wide Python install
:: directly (no bundled python\ folder here -- that only exists in the
:: PointTracker distribution folder).
:: =============================================================================

setlocal

set "APP_DIR=%~dp0"
set "PYTHON=C:\Users\ISN\AppData\Local\Python\pythoncore-3.14-64\python.exe"
set "TORCH_HOME=%APP_DIR%checkpoints\torch_cache"

echo.
echo  PrepByEKS ^| TAPIR Point Tracker v2  (dev)
echo.

if not exist "%PYTHON%" (
    echo  [ERROR] System Python not found at:
    echo    %PYTHON%
    echo.
    echo  Update the PYTHON path in this launch.bat if it has moved.
    pause
    exit /b 1
)

"%PYTHON%" -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Dependencies not installed for this Python.
    echo  Run install.bat first, then try again.
    pause
    exit /b 1
)

echo  Using: %PYTHON%
echo.

cd /d "%APP_DIR%"
"%PYTHON%" main.py %*

if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] App exited with code %errorlevel%.
    echo          Check the output above for details.
    pause
)

endlocal
