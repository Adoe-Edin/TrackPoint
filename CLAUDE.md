# PrepByEKS — TAPIR Point Tracker v2
## Claude Project Instructions & Technical Reference

**Client:** Institut Sukan Negara Malaysia (ISN)
**Work folder:** `C:\Users\ISN\Documents\Personal\Consult\App\Tapir-Tracker\`
**Python:** `C:\Users\ISN\AppData\Local\Python\pythoncore-3.14-64\python.exe`
**Launch:** `launch.bat` (double-click or run from terminal)

---

## Application Overview

A Windows desktop application for **biomechanical video point tracking** used by sports scientists at ISN. Users load a video, place tracking points on athletes, run AI inference (TAPIR or CoTracker), and review tracked trajectories with optional calibration and kinematics export.

**Stack:** Python 3.14 · Tkinter UI · OpenCV · PyTorch · tapnet (TAPIR) · CoTracker

---

## File Map

| File | Role |
|---|---|
| `main.py` | Full Tkinter application (~1920 lines) — all UI, state, playback, export |
| `tapir_engine.py` | TAPIR + CoTracker engines, `build_engine()`, GPU memory management |
| `config.py` | All tunable parameters (edit here first before touching engine/UI code) |
| `visualizer.py` | `draw_tracks()`, `draw_hud()`, `frames_to_preview()` |
| `calibration.py` | `LinearCalibration`, `FieldCalibration`, `CalibrationManager`, `compute_kinematics()` |
| `smoothing.py` | Savitzky-Golay smoothing + occlusion gap interpolation |
| `annotation_manager.py` | Per-frame annotation store (lines, arrows, rects, ellipses, text, angles) |
| `video_io.py` | `VideoReader` (downscales to DISPLAY_MAX), `VideoWriter`, `frames_to_rgb_tensor()` |
| `exporter.py` | `export_tracks_csv()`, `export_kinematics_csv()` |
| `checkpoints/tapir_checkpoint_panning.pt` | TAPIR weights (118.6 MB) — do not delete |
| `USER_MANUAL.pdf` | End-user manual (16 sections) |
| `launch.bat` | Launches `main.py` with the correct Python |
| `requirements.txt` | pip dependencies |

---

## Architecture

### Three-Tab UI (`main.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│  PrepByEKS | TAPIR Point Tracker v2   [Open Video] [Engine ▾]   │
│  [Run Tracking]  [Export]                                        │
├──────────┬──────────┬───────────────────────────────────────────┤
│  Setup   │  Replay  │  Calibrate                                 │
├──────────┴──────────┴───────────────────────────────────────────┤
│  (video canvas)          │  (sidebar panel)                      │
│                          │                                       │
│                          │  TRACKED POINTS / LINE CONNECTIONS    │
│                          │  (Setup)                              │
│                          │                                       │
│                          │  SMOOTH / KINEMATICS / PLAYBACK       │
│                          │  (Replay)                             │
│                          │                                       │
│                          │  LINEAR / FIELD CALIB / ASSIGN        │
│                          │  (Calibrate — scrollable sidebar)     │
└──────────────────────────┴───────────────────────────────────────┘
│  Status bar                                                       │
└───────────────────────────────────────────────────────────────────┘
```

**Setup tab** — place tracking query points (left-click), remove (right-click), rename, recolour, add line connections between points.

**Replay tab** — scrub/play result, smoothing controls (Savitzky-Golay window+poly), annotation toolbar (line/arrow/rect/ellipse/text/angle), trails toggle, kinematics export.

**Calibrate tab** — linear (2-point) or field (4-corner perspective) calibration; assign calibrations to tracked points; feeds `compute_kinematics()`.

### Engine selector (top bar)
`CoTracker | TAPIR | Auto` — dropdown. Auto tries TAPIR first, falls back to CoTracker.

---

## Engine Details (`tapir_engine.py`)

### TAPIR Engine

```python
TAPIR(pyramid_level=0, extra_convs=False, num_pips_iter=TAPIR_NUM_PIPS_ITER)
```

- `pyramid_level=0, extra_convs=False` — **must match** `tapir_checkpoint_panning.pt` exactly. The defaults (`pyramid_level=1, extra_convs=True`) produce a larger model that won't load from this checkpoint.
- Checkpoint is a **raw `OrderedDict`** (not wrapped in `{"model_state_dict": ...}`). Loaded with: `state = ckpt.get("model_state_dict", ckpt)`
- Input: `(B, T, H, W, C)` float32 `[0,1]`
- Output: `tracks (B, N, T, 2)` — must `.permute(1,0,2)` → `(T, N, 2)`
- Output: `occlusion (B, N, T)` logit — **higher = more occluded**. Visible score = `sigmoid(-logit)`.
- Query points format: `[t_abs, y_px, x_px]` where `t_abs` is the **absolute frame index** (not normalised 0–1), `y_px` / `x_px` are in **resized** frame pixel space.

#### JAX Import Bypass
`tapnet/__init__.py` imports JAX models on load (pulling in `einops`, `jax`, `flax`). Since only the PyTorch backend is used, `_import_tapir_model()` pre-stubs all JAX modules in `sys.modules` before importing, so `__init__.py` completes without loading any JAX code. Stubs are removed afterwards.

```python
_jax_stubs = ["einops", "tapnet.models", "tapnet.models.tapir_model",
              "tapnet.models.tapnet_model", "tapnet.robotap",
              "tapnet.robotap.tapir_clustering", "tapnet.tapvid",
              "tapnet.tapvid.evaluation_datasets"]
```

### CoTracker Engine

- Pre-resizes frames to `COTRACKER_INFER_H × COTRACKER_INFER_W` (216×384) before inference → ~7× faster than full 960×540. Tracks scaled back afterwards.
- Chunks long videos into `COTRACKER_CHUNK_FRAMES` segments with `COTRACKER_CHUNK_OVERLAP` overlap. Stitched with cosine-tapered blend weights.
- Chunk tensors explicitly deleted + `torch.cuda.empty_cache()` after each chunk to prevent VRAM accumulation.

### GPU Memory Management (critical)

After TAPIR inference, model weights (~200 MB) stay in the CUDA allocator if not explicitly freed. When switching to CoTracker, this causes OOM.

**Fix:** both engines have a `release()` method and `__del__`:
```python
def release(self):
    self.model.cpu()
    del self.model
    self.model = None
    torch.cuda.empty_cache()
```

`AppState.engine` stores the current engine. `build_engine(backend, previous_engine=self.state.engine)` releases it before loading the new one. **Always pass `previous_engine` when calling `build_engine()`.**

---

## Config Reference (`config.py`)

```python
# TAPIR inference resolution — 384 is the recommended balance
# 256 = fast but drifts; 512 = best but OOMs on <=8 GB VRAM
RESIZE_HEIGHT = 384
RESIZE_WIDTH  = 384

TAPIR_NUM_PIPS_ITER = 6      # refinement passes; 4=fast, 6=balanced, 8=best
OCCLUSION_THRESHOLD = 0.3    # visible score < this → occluded; lower = show more

COTRACKER_INFER_W = 384      # CoTracker inference width (pre-resize)
COTRACKER_INFER_H = 216      # CoTracker inference height
COTRACKER_CHUNK_FRAMES  = 150
COTRACKER_CHUNK_OVERLAP = 20

POINT_RADIUS = 2             # dot radius in pixels (drawn at 50% alpha)
TRAIL_ALPHA  = 0.6

DISPLAY_MAX_W = 960          # on-screen display resolution
DISPLAY_MAX_H = 540
```

---

## Key Bugs Resolved (do not re-introduce)

| # | Bug | Root cause | Fix |
|---|-----|-----------|-----|
| 1 | `tapnet.__spec__ is None` | Registering a bare `types.ModuleType` stub for `tapnet` gave it `__spec__=None` | Pre-stub JAX sub-modules instead; let tapnet itself import normally |
| 2 | `'model_state_dict'` KeyError | Checkpoint is a raw `OrderedDict`, not wrapped | `ckpt.get("model_state_dict", ckpt)` |
| 3 | Wrong model architecture | `TAPIR(pyramid_level=1, extra_convs=True)` defaults don't match checkpoint | `pyramid_level=0, extra_convs=False` |
| 4 | No tracking results shown | Query time `qf` was normalised `[0,1]`; TAPIR expects absolute frame index | `qf = float(self.state.query_frame)` |
| 5 | Wrong tracks shape | TAPIR returns `(B, N, T, 2)` not `(B, T, N, 2)` | `.permute(1, 0, 2)` after slicing batch |
| 6 | Points all visible / all hidden | Occlusion sign inverted: `sigmoid(+logit)` used but logit is occluded-positive | `visible = 1/(1 + exp(+logit))` (sigmoid of negated logit) |
| 7 | CUDA OOM on CoTracker | TAPIR weights not freed before CoTracker loads | `release()` + `empty_cache()` pattern |
| 8 | `cv2.resize` assertion zero dimension | `canvas.winfo_width()` returns 1 (not 0) before layout | Guard `if cw < 50: cw = CANVAS_MAX_W` |
| 9 | Points looked unchanged | Setup tab drew with hardcoded `POINT_RADIUS + 4`, no alpha — separate path from `draw_tracks()` | Both paths use `addWeighted(dot_layer, 0.5, frame, 0.5, 0, frame)` |
| 10 | CoTracker freeze on long videos | Full video tensor (e.g. 500 frames × 960×540) exhausts VRAM | Pre-resize to 384×216 + 150-frame chunks |

---

## Development Conventions

### File Writing (CRITICAL)
**Always use `Filesystem:write_file` for Python source files.** Never use PowerShell string injection (`Set-Content`, `.Replace()` multiline, `WriteAllBytes` with manual string ops) — PowerShell corrupts non-ASCII characters (e.g. `\x97` from em-dashes, curly quotes in f-strings) and silently fails on multiline anchors due to `\r\n` vs `\n` mismatches.

Workflow for edits:
1. Copy file to Claude's machine with `Filesystem:copy_file_user_to_claude`
2. Apply patches in Python (`str.replace()` on clean UTF-8 text)
3. Run `ast.parse()` to verify syntax
4. Write back with `Filesystem:write_file`

### PowerShell Usage
Safe for: `Get-Content`, `Select-String`, `Get-ChildItem`, `Remove-Item`, `Start-Process`, `Test-Path`

Avoid for: multiline string operations, file content modification, any use of `/s /b` flags (causes `PositionalParameterNotFound`)

### Python Patches
Always anchor on exact surrounding context, not just the target line. Verify with `assert OLD in src` before replacing. Run `ast.parse(result)` after every patch.

### venv
The app uses the system Python at `C:\Users\ISN\AppData\Local\Python\pythoncore-3.14-64\python.exe` directly (all packages installed globally to that interpreter). No venv for this project. Other apps in the same consult folder may use `venv` — do not cross-contaminate.

---

## Known Limitations / Future Work

- **TAPIR drift in team sports**: Players in matching kits look identical at the cost-volume level. Partially mitigated by `RESIZE=384` and `num_pips_iter=6`. Fundamental limitation of appearance-based tracking — cannot be fully fixed without identity re-ID.
- **CoTracker chunk stitching**: Overlapping blend weights smooth transitions but discontinuities may appear at boundaries on fast-moving points. Increasing `COTRACKER_CHUNK_OVERLAP` (e.g. to 30) helps.
- **No cancel button**: Long inference jobs (TAPIR on 300+ frame videos) cannot be interrupted without closing the app. A `threading.Event` stop signal would fix this.
- **Single query frame**: All points are queried at the same frame. Multi-frame query support (useful when different points enter frame at different times) is not yet implemented.
- **Export uses display resolution**: The exported video uses `DISPLAY_MAX_W/H` frames. If original resolution is needed, `VideoWriter` needs to re-read from disk rather than from `self.state.frames`.

---

## Folder Structure (clean state)

```
Tapir-Tracker/
├── CLAUDE.md                    ← this file (project instructions)
├── main.py                      ← main application
├── tapir_engine.py              ← inference engines
├── config.py                    ← all tunable parameters
├── visualizer.py                ← drawing utilities
├── calibration.py               ← calibration + kinematics
├── smoothing.py                 ← Savitzky-Golay + interpolation
├── annotation_manager.py        ← per-frame annotations
├── video_io.py                  ← video reading/writing
├── exporter.py                  ← CSV export
├── launch.bat                   ← app launcher
├── setup.bat                    ← first-time setup helper
├── install_deps.bat             ← pip install helper
├── requirements.txt             ← Python dependencies
├── README.md                    ← user-facing readme
├── HOW_TO_RUN.md                ← detailed run instructions
├── USER_MANUAL.pdf              ← full end-user manual
├── checkpoints/
│   └── tapir_checkpoint_panning.pt  ← TAPIR weights (118.6 MB, keep)
└── output/                      ← exported videos + CSVs (auto-created)
```

---

## Quick Diagnostics

**App won't start:**
```
cd C:\Users\ISN\Documents\Personal\Consult\App\Tapir-Tracker
C:\Users\ISN\AppData\Local\Python\pythoncore-3.14-64\python.exe main.py
```
Check the console output for import errors.

**TAPIR import fails:** Check that `tapnet` is installed:
```
python.exe -c "import importlib.util; print(importlib.util.find_spec('tapnet'))"
```

**CUDA OOM:** Lower `RESIZE_HEIGHT`/`RESIZE_WIDTH` in `config.py` (try 256). Or lower `COTRACKER_CHUNK_FRAMES` to 100.

**Points not visible after tracking:** Raise `OCCLUSION_THRESHOLD` to 0.5; lower it (0.2) if too many points disappear.

**Tracking drift:** Lower `TAPIR_NUM_PIPS_ITER` for speed, raise for accuracy (max 8). Ensure `RESIZE >= 384`.
