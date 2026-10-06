@echo off
:: =============================================================================
:: PrepByEKS -- TAPIR Point Tracker v2  --  INSTALLER
:: Log is written to install_log.txt in this folder.
::
:: Python is pre-bundled in the .\python\ folder (embeddable zip + tkinter).
:: This installer only handles pip packages -- no Python install needed.
:: =============================================================================

setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "LOG=%SCRIPT_DIR%install_log.txt"
set "FAILED=0"
set "PYTHON=%SCRIPT_DIR%python\python.exe"

echo. > "%LOG%"
echo ====================================================== >> "%LOG%"
echo   PrepByEKS TAPIR Point Tracker v2 -- INSTALLER LOG   >> "%LOG%"
echo   %DATE% %TIME%                                        >> "%LOG%"
echo ====================================================== >> "%LOG%"

echo.
echo  ======================================================
echo    PrepByEKS ^| TAPIR Point Tracker v2  --  INSTALLER
echo  ======================================================
echo.
echo  Log: %LOG%
echo.
echo  Press any key to begin ...
pause >nul

set "CHECKPOINT_DIR=%SCRIPT_DIR%checkpoints"
set "TORCH_CACHE=%CHECKPOINT_DIR%\torch_cache"
set "GETPIP_URL=https://bootstrap.pypa.io/get-pip.py"

call :LOG "SCRIPT_DIR = %SCRIPT_DIR%"
call :LOG "PYTHON     = %PYTHON%"

call :STEP0_PERMISSIONS
call :STEP1_PYTHON
if "%FAILED%"=="1" goto :ALLDONE
call :STEP2_PIP
if "%FAILED%"=="1" goto :ALLDONE
call :STEP3_TORCH
if "%FAILED%"=="1" goto :ALLDONE
call :STEP4_DEPS
if "%FAILED%"=="1" goto :ALLDONE
call :STEP5_TAPNET
call :STEP6_COTRACKER
call :STEP7_VERIFY

:ALLDONE
if "%FAILED%"=="1" (
    echo.
    echo  ============================================================
    echo    Installation did NOT complete. See install_log.txt
    echo  ============================================================
    echo.
    call :LOG "=== INSTALLATION FAILED ==="
) else (
    echo.
    echo  ============================================================
    echo    Installation complete!
    echo  ============================================================
    echo.
    echo    Double-click launch.bat to start the app.
    echo.
    call :LOG "=== INSTALLATION COMPLETE ==="
)
pause
endlocal
goto :EOF

:: =============================================================================
:: STEP 0 -- Fix read-only attributes on bundled python\ folder
:: Zips extracted on Windows often mark files as read-only.
:: pip cannot write into site-packages if files are read-only.
:: =============================================================================
:STEP0_PERMISSIONS
echo.
echo  [0/6]  Fixing folder permissions ...
call :LOG "STEP 0: fixing read-only attributes on python\ folder"
attrib -R "%SCRIPT_DIR%python\*.*" /S /D >nul 2>&1
attrib -R "%SCRIPT_DIR%checkpoints\*.*" /S /D >nul 2>&1
attrib -R "%SCRIPT_DIR%*.py" >nul 2>&1
echo         Done.
call :LOG "Permissions fixed."
goto :EOF

:: =============================================================================
:: STEP 1 -- Verify bundled Python
:: =============================================================================
:STEP1_PYTHON
echo.
echo  [1/6]  Python (bundled) ...
call :LOG "STEP 1: Python"

if not exist "%PYTHON%" (
    call :ERR "python\python.exe not found. The distribution folder may be incomplete."
    echo  Expected: %PYTHON%
    echo  Make sure you extracted the full zip including the python\ subfolder.
    set "FAILED=1"
    goto :EOF
)

"%PYTHON%" -c "import tkinter; import sys; print('Python', sys.version[:6], '+ tkinter OK')" >> "%LOG%" 2>&1
if not "%errorlevel%"=="0" (
    call :ERR "Bundled Python failed basic check. The python\ folder may be corrupted."
    set "FAILED=1"
    goto :EOF
)

echo         Bundled Python OK.
call :LOG "Bundled Python verified."
goto :EOF

:: =============================================================================
:: STEP 2 -- pip
:: =============================================================================
:STEP2_PIP
echo.
echo  [2/6]  pip ...
call :LOG "STEP 2: pip"

"%PYTHON%" -m pip --version >> "%LOG%" 2>&1
if "%errorlevel%"=="0" goto :STEP2_UPGRADE

echo         Bootstrapping pip ...
call :LOG "pip missing, bootstrapping"
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile '%TEMP%\get-pip.py' -UseBasicParsing"
if not "%errorlevel%"=="0" (
    call :ERR "Could not download get-pip.py. Check internet."
    set "FAILED=1"
    goto :EOF
)
"%PYTHON%" "%TEMP%\get-pip.py" >> "%LOG%" 2>&1
if not "%errorlevel%"=="0" (
    call :ERR "pip bootstrap failed."
    set "FAILED=1"
    goto :EOF
)
goto :STEP2_DONE

:STEP2_UPGRADE
"%PYTHON%" -m pip install --upgrade pip --quiet >> "%LOG%" 2>&1

:STEP2_DONE
echo         pip OK.
call :LOG "pip OK."
goto :EOF

:: =============================================================================
:: STEP 3 -- PyTorch
::
:: GPU architecture detection:
::   SM 12.x  = Blackwell  (RTX 50xx)  -> cu128 + torch>=2.7.0
::   SM 10.x  = Hopper     (H100 etc)  -> cu124 + torch>=2.6.0
::   SM  9.x  = Ada        (RTX 40xx)  -> cu124 + torch>=2.6.0
::   SM  8.x  = Ampere     (RTX 30xx)  -> cu124 + torch>=2.6.0
::   SM  7.x  = Volta/Turing(RTX 20xx) -> cu121 + torch>=2.6.0
::   No GPU                             -> cpu
::
:: nvidia-smi --query-gpu=compute_cap returns e.g. "12.0" or "8.6"
:: =============================================================================
:STEP3_TORCH
echo.
echo  [3/6]  PyTorch ...
call :LOG "STEP 3: PyTorch"

"%PYTHON%" -c "import torch" >> "%LOG%" 2>&1
if "%errorlevel%"=="0" (
    echo         Already installed -- skipping.
    echo         (Re-run install.bat with --reinstall-torch flag to force reinstall)
    call :LOG "torch already present, skipping."
    goto :EOF
)

set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_VER=torch>=2.6.0"
set "GPU_DESC=CPU-only"

nvidia-smi >nul 2>&1
if "%errorlevel%"=="0" (
    :: Read driver version
    for /f "skip=1 tokens=1" %%G in ('nvidia-smi --query-gpu^=driver_version --format^=csv 2^>nul') do (
        if not defined DRIVER_VER set "DRIVER_VER=%%G"
    )
    if not defined DRIVER_VER set "DRIVER_VER=unknown"

    :: Read compute capability (e.g. "12.0", "8.6", "7.5")
    for /f "skip=1 tokens=1" %%C in ('nvidia-smi --query-gpu^=compute_cap --format^=csv 2^>nul') do (
        if not defined COMPUTE_CAP set "COMPUTE_CAP=%%C"
    )
    if not defined COMPUTE_CAP set "COMPUTE_CAP=unknown"

    :: Extract major SM version
    for /f "tokens=1 delims=." %%M in ("!COMPUTE_CAP!") do set "SM_MAJOR=%%M"

    call :LOG "GPU driver=!DRIVER_VER! compute_cap=!COMPUTE_CAP! sm_major=!SM_MAJOR!"

    :: Select CUDA build based on SM major version
    if "!SM_MAJOR!"=="12" (
        :: Blackwell (RTX 50xx) -- requires cu128 + torch 2.7+
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
        set "TORCH_VER=torch>=2.7.0"
        set "GPU_DESC=CUDA 12.8 (Blackwell SM 12.x -- RTX 50xx)"
    ) else if "!SM_MAJOR!"=="9" (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "TORCH_VER=torch>=2.6.0"
        set "GPU_DESC=CUDA 12.4 (Ada/Hopper SM 9.x)"
    ) else if "!SM_MAJOR!"=="8" (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "TORCH_VER=torch>=2.6.0"
        set "GPU_DESC=CUDA 12.4 (Ampere SM 8.x -- RTX 30xx)"
    ) else if "!SM_MAJOR!"=="7" (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
        set "TORCH_VER=torch>=2.6.0"
        set "GPU_DESC=CUDA 12.1 (Volta/Turing SM 7.x -- RTX 20xx)"
    ) else (
        :: Unknown/very old -- try cu124 as best guess
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
        set "TORCH_VER=torch>=2.6.0"
        set "GPU_DESC=CUDA 12.4 (SM !COMPUTE_CAP! -- unknown architecture)"
    )

    echo         NVIDIA GPU detected.
    echo           Driver  : !DRIVER_VER!
    echo           GPU arch: SM !COMPUTE_CAP!
    echo           Build   : !GPU_DESC!
    call :LOG "Using index=!TORCH_INDEX! ver=!TORCH_VER!"
) else (
    echo         No GPU detected. Using CPU-only build.
    call :LOG "No GPU. index=!TORCH_INDEX!"
)

echo.
echo         Downloading PyTorch (!GPU_DESC!) ...
echo         CPU ~300 MB / CUDA ~2.5 GB -- do NOT close this window.
echo.
call :LOG "--- pip install torch begin (index=!TORCH_INDEX!) ---"

"%PYTHON%" -m pip install "!TORCH_VER!" "torchvision" ^
    --index-url "!TORCH_INDEX!" --timeout 300 --progress-bar on > "%TEMP%\pip_torch.txt" 2>&1
set "TORCH_EXIT=%errorlevel%"
type "%TEMP%\pip_torch.txt"
type "%TEMP%\pip_torch.txt" >> "%LOG%"
call :LOG "--- pip install torch end, exit=%TORCH_EXIT% ---"

if "%TORCH_EXIT%"=="0" (
    echo         PyTorch OK.
    call :LOG "PyTorch OK."
    goto :EOF
)
call :ERR "PyTorch FAILED (exit=%TORCH_EXIT%). Causes: internet drop, low disk (~3 GB needed), firewall."
set "FAILED=1"
goto :EOF

:: =============================================================================
:: STEP 4 -- Core deps
:: =============================================================================
:STEP4_DEPS
echo.
echo  [4/6]  Core dependencies ...
call :LOG "STEP 4: core deps"
call :LOG "--- pip install core deps begin ---"

"%PYTHON%" -m pip install "opencv-python>=4.8.0" "numpy>=1.24.0" "Pillow>=10.0.0" "scipy>=1.11.0" ^
    --timeout 120 --progress-bar on > "%TEMP%\pip_deps.txt" 2>&1
set "DEPS_EXIT=%errorlevel%"
type "%TEMP%\pip_deps.txt"
type "%TEMP%\pip_deps.txt" >> "%LOG%"
call :LOG "--- pip install core deps end, exit=%DEPS_EXIT% ---"

if "%DEPS_EXIT%"=="0" (
    echo         Core deps OK.
    call :LOG "Core deps OK."
    goto :EOF
)
call :ERR "Core deps FAILED (exit=%DEPS_EXIT%). See install_log.txt."
set "FAILED=1"
goto :EOF

:: =============================================================================
:: STEP 5 -- tapnet
:: =============================================================================
:STEP5_TAPNET
echo.
echo  [5/6]  tapnet (TAPIR backend) ...
call :LOG "STEP 5: tapnet"

git --version >nul 2>&1
if not "%errorlevel%"=="0" (
    echo         Git not found -- skipping tapnet.
    echo         Install Git: https://git-scm.com/download/win then re-run.
    echo         App will work with CoTracker only until then.
    call :LOG "Git not found, skipping tapnet."
    goto :EOF
)

"%PYTHON%" -c "from tapnet.torch import tapir_model" >nul 2>&1
if "%errorlevel%"=="0" (
    echo         Already installed -- skipping.
    call :LOG "tapnet already present."
    goto :STEP5_CHECKPOINT
)

call :LOG "--- pip install tapnet begin ---"
"%PYTHON%" -m pip install "git+https://github.com/google-deepmind/tapnet.git" ^
    --timeout 180 --progress-bar on > "%TEMP%\pip_tapnet.txt" 2>&1
set "TAPNET_EXIT=%errorlevel%"
type "%TEMP%\pip_tapnet.txt"
type "%TEMP%\pip_tapnet.txt" >> "%LOG%"
call :LOG "--- pip install tapnet end, exit=%TAPNET_EXIT% ---"

if "%TAPNET_EXIT%"=="0" (
    echo         tapnet OK.
    call :LOG "tapnet OK."
) else (
    echo         [WARN] tapnet failed -- CoTracker still works.
    call :LOG "tapnet failed (non-fatal)."
)

:STEP5_CHECKPOINT
if exist "%CHECKPOINT_DIR%\tapir_checkpoint_panning.pt" (
    call :LOG "TAPIR checkpoint present."
) else (
    echo         [WARN] TAPIR checkpoint not found in checkpoints\
    call :LOG "TAPIR checkpoint missing."
)
goto :EOF

:: =============================================================================
:: STEP 6 -- CoTracker weights
:: =============================================================================
:STEP6_COTRACKER
echo.
echo  [6/6]  CoTracker weights (~300 MB) -- do not close this window ...
call :LOG "STEP 6: CoTracker weights"

if not exist "%TORCH_CACHE%" mkdir "%TORCH_CACHE%"

"%PYTHON%" -c "import os; os.environ['TORCH_HOME']=r'%TORCH_CACHE%'; import torch; torch.hub.load('facebookresearch/co-tracker','cotracker3_offline',trust_repo=True); print('CoTracker OK')" > "%TEMP%\cotracker.txt" 2>&1
set "CT_EXIT=%errorlevel%"
type "%TEMP%\cotracker.txt"
type "%TEMP%\cotracker.txt" >> "%LOG%"
call :LOG "CoTracker exit=%CT_EXIT%"

if "%CT_EXIT%"=="0" (
    echo         CoTracker weights cached OK.
    call :LOG "CoTracker OK."
) else (
    echo         [WARN] CoTracker cache failed -- app will download at first launch.
    call :LOG "CoTracker pre-cache failed (non-fatal)."
)
goto :EOF

:: =============================================================================
:: STEP 7 -- Verification
:: =============================================================================
:STEP7_VERIFY
echo.
echo  Verifying ...
call :LOG "--- Verification ---"

"%PYTHON%" -c "import tkinter, torch, cv2, numpy, PIL, scipy; print('  tkinter: OK'); print('  torch  :', torch.__version__); print('  opencv :', cv2.__version__); print('  numpy  :', numpy.__version__); print('  CUDA   :', torch.cuda.is_available()); print('  GPU    :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" > "%TEMP%\pip_verify.txt" 2>&1
set "VER_EXIT=%errorlevel%"
type "%TEMP%\pip_verify.txt"
type "%TEMP%\pip_verify.txt" >> "%LOG%"
call :LOG "Verification exit=%VER_EXIT%"

if "%VER_EXIT%"=="0" goto :EOF
    call :ERR "Verification failed -- see install_log.txt."
    set "FAILED=1"
goto :EOF

:: =============================================================================
:: Helpers
:: =============================================================================
:LOG
echo   [LOG] %~1 >> "%LOG%"
goto :EOF

:ERR
echo.
echo  [ERROR]  %~1
echo  [ERROR]  %~1 >> "%LOG%"
goto :EOF
