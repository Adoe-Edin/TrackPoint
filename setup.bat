@echo off
:: =============================================================================
:: PrepByEKS -- TAPIR Point Tracker v2
:: Windows setup -- installs into Anaconda base explicitly
:: =============================================================================

set CONDA_PYTHON=C:\ProgramData\Anaconda3\python.exe
set CONDA_PIP=C:\ProgramData\Anaconda3\Scripts\pip.exe

echo.
echo  PrepByEKS ^| TAPIR Point Tracker v2 -- Setup
echo  ==============================================
echo  Using Python: %CONDA_PYTHON%
echo.

echo [1/3] Installing core dependencies (PyTorch cu128 + opencv + scipy)...
"%CONDA_PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
"%CONDA_PIP%" install opencv-python numpy Pillow scipy

echo.
echo [2/3] Installing CoTracker from GitHub (fallback backend)...
"%CONDA_PIP%" install "git+https://github.com/facebookresearch/co-tracker.git"

echo.
set /p INSTALL_TAPIR="[3/3] Install TAPIR primary backend? Requires git. [y/N]: "
if /i "%INSTALL_TAPIR%"=="y" (
    "%CONDA_PIP%" install "git+https://github.com/google-deepmind/tapnet.git"
    echo.
    echo  Download checkpoint:
    echo    URL : https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt
    echo    Save: checkpoints\tapir_checkpoint_panning.pt
    mkdir checkpoints 2>nul
)

echo.
echo  Setup complete. Launch with:
echo    launch.bat
echo    launch.bat --video "C:\path\to\video.mp4"
echo.
pause
