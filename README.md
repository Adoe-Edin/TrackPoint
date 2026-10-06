# PrepByEKS — TAPIR Point Tracker v2
**Institut Sukan Negara Malaysia | BPS Division**

Track any user-selected point through video using TAPIR (DeepMind) or
CoTracker (Meta) as fallback.  Points can represent anything —
anatomical landmarks, equipment, markers, or any arbitrary object.

---

## Quick Start

```powershell
cd "C:\Users\ISN\Documents\Personal\Consult\App\Tapir-Tracker"
setup.bat          # run once — installs dependencies
python main.py
```

---

## Workflow

### Setup Tab
| Action | How |
|--------|-----|
| Load video | Open Video button |
| Place point | Click anywhere on the frame — auto-labelled P1, P2, P3... |
| Rename point | Double-click list row **or** Rename button |
| Change color | Select row → Color button |
| Remove point | Right-click canvas (nearest) or select row → Remove |
| Add line connection | Choose From / To, set label + color + solid/dashed → Add Connection |
| Fan out from one point | From stays selected after each Add — just change To |

### Replay Tab
- Annotated playback with smoothed tracks, trails, line connections, HUD
- Scrubber, |< << >> >| seek buttons, adjustable speed (0.25x – 4x)

#### Annotation Toolbar
| Tool | Action |
|------|--------|
| Select | Disable drawing |
| Line | Click-drag line |
| Arrow | Click-drag arrow |
| Rect | Click-drag rectangle |
| Ellipse | Click-drag ellipse |
| Text | Click → type label |
| Angle | Three clicks → auto-shows degrees |
| Undo / Clear Frame / Clear All | History management |

Toggle **All frames** to apply an annotation globally across every frame.

#### Smoothing Panel
| Control | Effect |
|---------|--------|
| Enabled | Toggle Savitzky-Golay on/off |
| Window | Kernel size (odd integer — larger = smoother) |
| Poly | Polynomial order (lower = smoother) |
| Apply | Re-run with current settings |
| Trails / Labels | Show/hide trails and point labels |

---

## Output  (`output/`)

| File | Content |
|------|---------|
| `<name>_tapir_tracked.mp4` | Annotated video |
| `<name>_tracks_<timestamp>.csv` | Per-frame x, y, visibility, score |

CSV columns: `frame, time_s, point_id, label, x_px, y_px, visible, score`

---

## File Structure

```
Tapir-Tracker/
├── main.py               ← Full application
├── tapir_engine.py       ← TAPIR / CoTracker inference
├── smoothing.py          ← Occlusion interpolation + Savitzky-Golay
├── annotation_manager.py ← Per-frame annotation store + renderer
├── visualizer.py         ← Trails, line connections, dashed lines, HUD
├── video_io.py           ← VideoReader / VideoWriter
├── exporter.py           ← CSV export
├── config.py             ← All tuneable parameters
├── setup.bat             ← One-click Windows installer
├── requirements.txt      ← pip dependencies
├── checkpoints/          ← Place tapir_checkpoint_panning.pt here
└── output/               ← Auto-created on first export
```

---

## Backends

| Backend | Install | Checkpoint |
|---------|---------|------------|
| **TAPIR** (primary) | `pip install git+https://github.com/google-deepmind/tapnet.git` | [tapir_checkpoint_panning.pt](https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt) → `checkpoints/` |
| **CoTracker** (fallback) | `pip install cotracker` | None — auto-downloads |

Engine auto-falls back to CoTracker if tapnet is not installed.

---

*PrepByEKS · Edin Kardin Suwarganda · ISN BPS Division*
