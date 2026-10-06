@echo off
:: =============================================================================
:: PrepByEKS -- TAPIR Point Tracker v2
:: Install / repair dependencies using Python 3.14 (where torch lives)
:: =============================================================================

set PYTHON=C:\Users\ISN\AppData\Local\Python\pythoncore-3.14-64\python.exe
set PIP=%PYTHON% -m pip

echo.
echo  PrepByEKS ^| Dependency installer
echo  Using: %PYTHON%
echo.

echo [1] torch + torchvision (cu128)...
%PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --timeout 120

echo.
echo [2] opencv-python...
%PIP% install opencv-python --timeout 60

echo.
echo [3] numpy Pillow scipy...
%PIP% install numpy Pillow scipy --timeout 60

echo.
echo [4] CoTracker...
%PIP% install "git+https://github.com/facebookresearch/co-tracker.git" --timeout 120

echo.
echo [5] Verifying...
"%PYTHON%" -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
"%PYTHON%" -c "import torchvision; print('torchvision:', torchvision.__version__)"
"%PYTHON%" -c "import cv2; print('opencv:', cv2.__version__)"
"%PYTHON%" -c "import cotracker; print('cotracker: OK')"
"%PYTHON%" -c "import scipy, PIL, numpy; print('scipy/Pillow/numpy: OK')"

echo.
echo  Done. Run: launch.bat
echo.
pause
