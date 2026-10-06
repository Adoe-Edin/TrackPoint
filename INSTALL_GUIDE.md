# PrepByEKS — TAPIR Point Tracker v2
## Installation Guide for New Machines

---

## Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10 or 11 (64-bit) |
| Internet | Required during install (~2–3 GB download) |
| GPU | NVIDIA recommended. AMD/Intel CPU-only also works (slower) |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~5 GB free for Python + PyTorch + weights |
| Git | Required for TAPIR backend (https://git-scm.com/download/win) |

---

## What You Receive

```
Tapir-Tracker/
├── install.bat                  <- Run this ONCE on a new machine
├── launch.bat                   <- Run this every time to start the app
├── INSTALL_GUIDE.md             <- This file
├── app source files (*.py)
├── checkpoints/
│   └── tapir_checkpoint_panning.pt   <- TAPIR weights (118 MB)
└── output/                      <- Exported videos and CSVs go here
```

---

## Step 1 — Install Git (if not already installed)

Git is needed to install the TAPIR backend from GitHub.

1. Go to: https://git-scm.com/download/win
2. Download and run the installer (all defaults are fine)
3. Confirm with: open a Command Prompt and type `git --version`

You can skip this step if you only want to use the CoTracker engine.

---

## Step 2 — Run install.bat

1. Right-click `install.bat` → **Run as administrator** is NOT needed
2. Simply **double-click** it to open a command window
3. The installer will:
   - Download Python 3.13 into a local `python\` subfolder (no system changes)
   - Detect whether you have an NVIDIA GPU and install the matching PyTorch build
   - Install all other dependencies (OpenCV, NumPy, SciPy, Pillow)
   - Install tapnet (TAPIR engine) from GitHub
   - Pre-download CoTracker weights into `checkpoints\torch_cache\`
   - Run a quick verification test

**Expected time:** 5–15 minutes depending on internet speed and GPU build size.

**Expected download size:**
- NVIDIA GPU (CUDA build): ~2.5 GB
- No GPU (CPU-only build): ~400 MB

When you see `Installation complete!` the setup is done.

---

## Step 3 — Copy the TAPIR checkpoint

The TAPIR checkpoint file (`tapir_checkpoint_panning.pt`, 118 MB) must be
placed in the `checkpoints\` folder. It is not downloaded automatically.

Obtain the file from the PrepByEKS developer (Edin) or from the shared ISN
drive, then copy it to:

```
Tapir-Tracker\checkpoints\tapir_checkpoint_panning.pt
```

The app will still work with CoTracker if this file is missing, but TAPIR
tracking will not be available until the checkpoint is present.

---

## Step 4 — Launch the app

**Double-click `launch.bat`** to start the app.

A console window will open briefly while Python loads, then the main
application window will appear.

---

## Troubleshooting

### "No module named X" on launch
Re-run `install.bat`. A package may have failed silently.

### CUDA out of memory error during tracking
Open `config.py` and reduce:
```
RESIZE_HEIGHT = 256
RESIZE_WIDTH  = 256
```
Or reduce `COTRACKER_CHUNK_FRAMES = 100`.

### "nvidia-smi not found" but I have a GPU
Update your NVIDIA drivers. Download from: https://www.nvidia.com/drivers

### CUDA 12.1 build crashes but I have an older GPU
Open a Command Prompt in the `Tapir-Tracker\` folder and run:
```
python\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
This installs the CUDA 11.8 build, which supports older drivers (>= 452).

### tapnet failed to install
1. Confirm Git is installed: `git --version` in a Command Prompt
2. Re-run install.bat, or manually run:
   ```
   python\python.exe -m pip install git+https://github.com/google-deepmind/tapnet.git
   ```

### CoTracker fails on first tracking run (weights not found)
The weight pre-cache in Step 6 of install.bat may have failed. The app
will try to download them automatically at runtime. Ensure internet access
is available on the machine.

---

## Uninstalling

To remove the app completely:
1. Delete the `Tapir-Tracker\` folder
2. No system-level changes were made — Python was installed locally and
   nothing was added to the Windows registry or system PATH.

---

## Machine-to-Machine Transfer (No Internet)

If the target machine has no internet access, you can transfer a pre-built
installation:

1. On the source machine (internet access), run install.bat to completion
2. Copy the entire `Tapir-Tracker\` folder to the target machine
   (this includes `python\`, `checkpoints\`, all source files)
3. On the target machine, double-click `launch.bat` — no install needed

The `python\` folder contains a self-contained Python interpreter with all
packages pre-installed. Total folder size after install: ~5–6 GB.

**Note:** The CUDA build inside `python\Lib\site-packages\torch\lib\` is
compiled for a specific CUDA runtime version (12.1 by default). If the
target machine has an older NVIDIA driver (< 527), PyTorch will automatically
fall back to CPU mode. Tracking still works, just slower.
