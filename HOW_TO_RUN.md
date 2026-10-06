# PrepByEKS — TAPIR Point Tracker v2
## How to Run

---

## Quick Launch

Double-click **`launch.bat`** in the app folder, or from any terminal:

```bat
cd "C:\Users\ISN\Documents\Personal\Consult\App\Tapir-Tracker"
launch.bat
```

To open a video immediately on launch:

```bat
launch.bat --video "C:\path\to\your\video.mp4"
```

---

## Step-by-Step Workflow

### 1 — Open Video (Setup Tab)
- Click **Open Video** in the top bar
- Supported formats: `.mp4` `.avi` `.mov` `.mkv`
- Frames are loaded at reduced display resolution (960×540) to save memory
- Status bar shows: frame count, FPS, original resolution, display resolution

---

### 2 — Place Tracking Points (Setup Tab)

| Action | How |
|--------|-----|
| Place a point | **Left-click** anywhere on the frame |
| Points are auto-labelled | P1, P2, P3… |
| Rename a point | **Double-click** its row in the list, or select + click **Rename** |
| Change point colour | Select row → **Color** |
| Remove nearest point | **Right-click** on the canvas |
| Remove selected point | Select row → **Remove** |
| Clear all | **Clear All** button |

> Points represent anything — anatomical landmarks, equipment, markers, reference objects.

---

### 3 — Define Line Connections (Setup Tab, right panel)

Optional: draw lines between tracked points that appear on replay.

1. Choose **From** and **To** points from the dropdowns
2. Optionally set a label, colour, and solid/dashed style
3. Click **+ Add Connection**
4. The **From** point stays selected — change only **To** to fan multiple lines from one point
5. A duplicate warning appears if the same pair is added twice

---

### 4 — Run Tracking (any tab)

- Click **Run Tracking** in the top bar
- CoTracker downloads its pretrained weights (~200 MB) on first run — wait for this
- Progress bar shows inference progress
- App automatically switches to the **Replay** tab when done

---

### 5 — Review in Replay Tab

**Playback controls:**

| Control | Action |
|---------|--------|
| **Play / Pause** | Start/stop video playback |
| **\|<  <<  >>  >\|** | Jump to start / back 10 / forward 10 / end |
| **Scrubber** | Drag to any frame |
| **Speed** | 0.25× – 4× playback speed |
| **A key** | Previous frame |
| **D key** | Next frame |
| **Scroll wheel** | Previous / next frame |

**Smoothing controls (bottom left):**

| Control | Effect |
|---------|--------|
| **On** checkbox | Toggle Savitzky-Golay smoothing |
| **Win** | Kernel size — larger = smoother trajectory |
| **Poly** | Polynomial order — lower = smoother |
| **Apply** | Re-run smoothing with current settings |
| **Trails** | Show/hide motion trails |
| **Labels** | Show/hide point labels |

**Annotation toolbar:**

| Tool | How to use |
|------|-----------|
| Select | Disable drawing |
| Line | Click-drag |
| Arrow | Click-drag |
| Rect | Click-drag |
| Ellipse | Click-drag |
| Text | Click → type |
| Angle | Click 3 times (shows degrees automatically) |
| All frames | Toggle to apply annotation to every frame |
| Undo | Remove last annotation |
| Clear Frame / Clear All | Wipe annotations |

---

### 6 — Calibrate (Calibrate Tab)

Required before computing kinematics.

#### Linear Calibration (2 points)
1. Select **Linear (2 pts)** mode
2. Click 2 points on a known distance in the frame (e.g. lane markings, ruler)
3. Enter the real-world distance in metres (e.g. `1.00`)
4. Give it a name, click **+ Add Linear Calibration**
5. Repeat for more lines if needed

#### Field Calibration (4 points — perspective-corrected)
1. Select **Field (4 pts)** mode
2. Click the 4 corners of a known rectangular area in order:
   **Top-Left → Top-Right → Bottom-Right → Bottom-Left**
3. Enter real-world **W (m)** and **H (m)**
4. Give it a name, click **+ Add Field Calibration**

> Right-click on the canvas to undo the last placed point.
> The sidebar scrolls if content overflows — use the scrollbar or mouse wheel.

#### Assign Calibrations to Points
1. In **Assign to Points**, choose a tracked point and a calibration
2. Click **Assign**
3. One point can use Linear, another can use Field — mix freely
4. Assignments show in the list below; select + **Clear Assignment** to remove

---

### 7 — Compute Kinematics (Replay Tab, bottom right)

After calibration and assignment:

1. Click **Compute & Export** in the kinematics section
2. Adjust **Win** and **Poly** spinners if needed (same Savitzky-Golay settings)
3. A CSV is saved automatically to the `output/` folder

**Output columns:**

| Column | Unit | Description |
|--------|------|-------------|
| `world_x_m` / `world_y_m` | m | Real-world position |
| `vel_x_ms` / `vel_y_ms` | m/s | Velocity components |
| `speed_ms` | m/s | Resultant speed |
| `accel_ms2` | m/s² | Tangential acceleration |
| `distance_m` | m | Cumulative arc-length |

---

### 8 — Export

Click **Export** in the top bar to save:

| File | Location | Content |
|------|----------|---------|
| `<video>_tapir_tracked.mp4` | `output/` | Annotated video with tracks, lines, HUD |
| `<video>_tracks_<timestamp>.csv` | `output/` | Pixel coordinates per frame per point |
| `<video>_kinematics_<timestamp>.csv` | `output/` | World-space kinematics (if computed) |

---

## Output Folder

All exports go to:
```
C:\Users\ISN\Documents\Personal\Consult\App\Tapir-Tracker\output\
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'torch'` | Run `launch.bat` — it uses Python 3.14 explicitly |
| App opens but crashes on video load | Lower `DISPLAY_MAX_W` / `DISPLAY_MAX_H` in `config.py` |
| Tracking is poor | CoTracker may not have downloaded weights yet — check internet, try again |
| Points disappear too often | Lower `OCCLUSION_THRESHOLD` in `config.py` (try `0.2`) |
| Slow tracking | Reduce `COTRACKER_WINDOW_SIZE` in `config.py` (try `4`) |
| Kinematics all zeros | Check that points are assigned to calibrations in the Calibrate tab |
| Need to reinstall dependencies | Run `install_deps.bat` from the app folder |

---

## Config Knobs (`config.py`)

```python
DISPLAY_MAX_W        = 960    # Display resolution width  (lower = less RAM)
DISPLAY_MAX_H        = 540    # Display resolution height
OCCLUSION_THRESHOLD  = 0.3    # Lower = show more points through occlusion
COTRACKER_WINDOW_SIZE= 8      # Higher = more temporal context (slower)
SMOOTH_WINDOW_DEFAULT= 11     # Default smoothing kernel
SMOOTH_POLYORDER_DEFAULT = 3  # Default polynomial order
MAX_POINTS           = 30     # Maximum tracked points per session
TRAIL_LENGTH         = 60     # Frames of trail history shown
```

---

*PrepByEKS · Edin Kardin Suwarganda · ISN BPS Division*
