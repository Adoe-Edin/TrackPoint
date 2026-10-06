# =============================================================================
# TAPIR Point Tracker v2 — Configuration
# PrepByEKS | Institut Sukan Negara Malaysia
# =============================================================================

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
# Write output to a user-writable location.
# If the app is installed in a protected folder (e.g. C:\Program Files\),
# fall back to Documents\PointTracker\output so exports always succeed.
def _get_output_dir():
    _app_output = os.path.join(BASE_DIR, "output")
    try:
        os.makedirs(_app_output, exist_ok=True)
        _test = os.path.join(_app_output, ".write_test")
        open(_test, "w").close()
        os.remove(_test)
        return _app_output
    except (PermissionError, OSError):
        _user_output = os.path.join(os.path.expanduser("~"), "Documents",
                                    "PointTracker", "output")
        os.makedirs(_user_output, exist_ok=True)
        return _user_output

OUTPUT_DIR = _get_output_dir()
CHECKPOINT_DIR   = os.path.join(BASE_DIR, "checkpoints")
TAPIR_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "tapir_checkpoint_panning.pt")

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_BACKEND = "tapir"

# TAPIR inference resolution.
# 256 = fast, drifts on crowded scenes.
# 384 = good balance of quality vs VRAM (recommended — fits most GPUs).
# 512 = best quality but often OOMs on <=8GB VRAM cards.
RESIZE_HEIGHT = 384
RESIZE_WIDTH  = 384
CHUNK_SIZE    = 32

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE          = "cuda"
MIXED_PRECISION = True

# ── TAPIR tracking quality ────────────────────────────────────────────────────
# PIPs refinement passes per pyramid level.
# 4 = fast (default).  6 = balanced.  8 = best, slowest.
TAPIR_NUM_PIPS_ITER = 6

# ── Tracking ──────────────────────────────────────────────────────────────────
MAX_POINTS   = 30
TRAIL_LENGTH = 60

# Visibility score below this = point considered occluded.
# 0.3 = permissive (show more points).  0.5 = strict.
OCCLUSION_THRESHOLD = 0.3

# ── CoTracker settings ────────────────────────────────────────────────────────
# Temporal window. Larger = more context, slower. Default 8.
COTRACKER_WINDOW_SIZE = 8

# Inference resolution for CoTracker (~7x faster than full 960x540).
COTRACKER_INFER_W = 384
COTRACKER_INFER_H = 216

# Chunk processing to prevent OOM / freeze on long videos.
COTRACKER_CHUNK_FRAMES  = 150
COTRACKER_CHUNK_OVERLAP = 20

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTH_DEFAULT_ENABLED   = True
SMOOTH_WINDOW_DEFAULT    = 11
SMOOTH_POLYORDER_DEFAULT = 3
INTERP_OCCLUSIONS        = True

# ── Visualisation ─────────────────────────────────────────────────────────────
# Tracking dot radius — 1/3 of original (was 5 → now 2), drawn at 50% alpha.
POINT_RADIUS = 2
TRAIL_ALPHA  = 0.6

# ── Display resolution ────────────────────────────────────────────────────────
DISPLAY_MAX_W = 960
DISPLAY_MAX_H = 540

# 30 distinct colors (BGR)
COLORS = [
    (0,   255, 128), (0,   200, 255), (255, 100,   0), (180,   0, 255),
    (255, 220,   0), (0,   128, 255), (0,   255,   0), (255,  50, 150),
    (50,  255, 200), (255, 160,  30), (100, 255, 255), (255,   0, 100),
    (0,   180, 180), (200, 255,   0), (128,   0, 255), (255, 128,   0),
    (0,   255, 200), (200,   0, 200), (255, 200,   0), (0,   100, 255),
    (150, 255,   0), (255,   0, 200), (0,   220, 100), (100, 100, 255),
    (255,  80,  80), (80,  255, 160), (200, 160,   0), (0,   160, 200),
    (255, 140, 200), (140, 200, 255),
]

LINE_COLORS = [
    (255, 255, 255), (0, 255, 255), (255, 165, 0),
    (0, 255, 0),     (255, 0, 255), (0, 200, 255),
    (200, 200, 0),   (0, 180, 255), (255, 100, 100),
]

# ── Annotation ────────────────────────────────────────────────────────────────
ANNOT_DEFAULT_COLOR     = "#00c896"
ANNOT_DEFAULT_WIDTH     = 2
ANNOT_DEFAULT_FONT_SIZE = 14
ANNOT_ARROW_TIP_LEN     = 15

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE    = "PrepByEKS — TAPIR Point Tracker"
CANVAS_MAX_W = DISPLAY_MAX_W
CANVAS_MAX_H = DISPLAY_MAX_H
BG_COLOR     = "#0d0d0d"
ACCENT_COLOR = "#00c896"
TEXT_COLOR   = "#e8e8e8"
BTN_COLOR    = "#1a1a2e"
PANEL_BG     = "#111827"
SIDEBAR_W    = 290

# ── Export ────────────────────────────────────────────────────────────────────
EXPORT_VIDEO  = True
EXPORT_CSV    = True
VIDEO_CODEC   = "mp4v"
CSV_DELIMITER = ","
