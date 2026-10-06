# =============================================================================
# TAPIR Point Tracker v2 — Main Application
# PrepByEKS | Institut Sukan Negara Malaysia
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import threading
import time as _time
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import config
from annotation_manager import (
    AnnotAngle, AnnotArrow, AnnotEllipse, AnnotLine, AnnotRect, AnnotText,
    AnnotationManager, draw_annotations, hex_to_bgr,
)
from config import (
    ACCENT_COLOR, APP_TITLE, BG_COLOR, BTN_COLOR, CANVAS_MAX_H, CANVAS_MAX_W,
    COLORS, LINE_COLORS, MAX_POINTS, PANEL_BG, SIDEBAR_W, TEXT_COLOR,
    SMOOTH_DEFAULT_ENABLED, SMOOTH_WINDOW_DEFAULT, SMOOTH_POLYORDER_DEFAULT,
)
from exporter import export_tracks_csv, export_kinematics_csv
from calibration import (
    CalibrationManager, LinearCalibration, FieldCalibration, compute_kinematics,
)
from smoothing import process_tracks
from tapir_engine import TrackResult, build_engine
from video_io import VideoReader, VideoWriter, frames_to_rgb_tensor
from visualizer import LineConnection, draw_hud, draw_tracks, frames_to_preview
from face_id_core import FaceManager
from face_id_tab_tapir import FaceIDTab, make_name_label, athlete_store, grab_frame_tapir
APP_ROOT = Path(__file__).parent.parent

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


# =============================================================================
# State
# =============================================================================

class AppState:
    def __init__(self):
        self.video:            VideoReader = None
        self.frames:           list        = []
        self.result:           TrackResult = None
        self.smoothed_tracks:  np.ndarray  = None
        self.query_frame:      int  = 0
        self.query_points:     list = []
        self.point_labels:     list = []
        self.point_colors:     list = []
        self.tracking_done:    bool = False
        self.engine:           object = None   # current inference engine (holds GPU memory)
        self.trim_source_path: str   = ""
        self.trim_total_fr:    int   = 0
        self.trim_fps:         float = 30.0
        self.connections:      list = []
        self.current_frame:    int  = 0
        self.is_playing:       bool = False
        self.smooth_enabled:   bool = SMOOTH_DEFAULT_ENABLED
        self.smooth_window:    int  = SMOOTH_WINDOW_DEFAULT
        self.smooth_polyorder: int  = SMOOTH_POLYORDER_DEFAULT
        self.annot_manager:    AnnotationManager = AnnotationManager()
        self.setup_scale: float = 1.0
        self.setup_ox:    int   = 0
        self.setup_oy:    int   = 0
        self.setup_fw:    int   = 1
        self.setup_fh:    int   = 1
        self.replay_scale: float = 1.0
        self.replay_ox:    int   = 0
        self.replay_oy:    int   = 0
        self.replay_fw:    int   = 1
        self.replay_fh:    int   = 1
        # Calibration
        self.calib_mgr: CalibrationManager = CalibrationManager()
        # Calibrate-tab display geometry
        self.calib_scale: float = 1.0
        self.calib_ox:    int   = 0
        self.calib_oy:    int   = 0
        self.calib_fw:    int   = 1
        self.calib_fh:    int   = 1


# =============================================================================
# Helpers
# =============================================================================

def _make_btn(parent, text, cmd, bg=BTN_COLOR, fg=TEXT_COLOR,
              font=("Segoe UI", 9), side=tk.LEFT, padx=4, **kw):
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                  relief=tk.FLAT, padx=10, pady=4, font=font,
                  activebackground=ACCENT_COLOR, cursor="hand2", **kw)
    b.pack(side=side, padx=padx, pady=2)
    return b


def bgr_to_hex(bgr: tuple) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# =============================================================================
# Application
# =============================================================================

class TAPIRApp:

    def __init__(self, root: tk.Tk, initial_video: str = None):
        self.root   = root
        self.state  = AppState()
        self._after_id = None
        self._annot_tool     = "select"
        self._annot_color    = config.ANNOT_DEFAULT_COLOR
        self._annot_width    = config.ANNOT_DEFAULT_WIDTH
        self._annot_global   = False
        self._draw_start     = None
        self._draw_angle_pts = []
        self._temp_line_id   = None
        # Calibration tool state
        self._calib_mode      = "linear"   # "linear" | "field"
        self._calib_click_pts = []         # list of (xn,yn) normalised pending
        # Magnifier state
        self._magnifier_on    = False
        # Trim tab widget/playback state
        self._trim_frame_bgr  = None
        self._trim_photo      = None
        self._trim_drag       = None
        self._trim_playing    = False
        self._trim_play_cap   = None
        self._trim_start_pct  = tk.DoubleVar(value=0.0)
        self._trim_end_pct    = tk.DoubleVar(value=100.0)
        self._trim_play_pos   = tk.DoubleVar(value=0.0)
        self._setup_mag_img   = None       # keep PhotoImage refs alive
        self._calib_mag_img   = None

        # Face ID
        self.face_mgr = FaceManager(on_status=self._status)
        self._face_id_tab = None   # set inside _build_ui after notebook exists
        self._build_ui()

        if initial_video:
            self._load_video(initial_video)

    # =========================================================================
    # UI Construction
    # =========================================================================

    def _build_ui(self):
        self.root.title(APP_TITLE)
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(True, True)

        top = tk.Frame(self.root, bg="#0a0a0a", pady=5)
        top.pack(side=tk.TOP, fill=tk.X)
        tk.Label(top, text="PrepByEKS  |  TAPIR Point Tracker  v2",
                 fg=ACCENT_COLOR, bg="#0a0a0a",
                 font=("Consolas", 12, "bold")).pack(side=tk.LEFT, padx=14)
        _make_btn(top, "Open Video",   self._open_video,   side=tk.LEFT)

        # Engine selector
        self._engine_var = tk.StringVar(value="CoTracker")
        engine_frame = tk.Frame(top, bg="#0a0a0a")
        engine_frame.pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(engine_frame, text="Engine:", fg="#888", bg="#0a0a0a",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._engine_cb = ttk.Combobox(
            engine_frame, textvariable=self._engine_var,
            values=["CoTracker", "TAPIR", "Auto"],
            width=10, state="readonly", font=("Segoe UI", 8))
        self._engine_cb.pack(side=tk.LEFT, padx=4)
        self._engine_cb.bind("<<ComboboxSelected>>", self._on_engine_change)
        _make_btn(top, "Run Tracking", self._run_tracking, side=tk.LEFT)
        _make_btn(top, "Export",       self._export,       side=tk.LEFT)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("Dark.TNotebook.Tab",
                        background="#1a1a2e", foreground=TEXT_COLOR,
                        padding=[14, 6], font=("Segoe UI", 9))
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", PANEL_BG)],
                  foreground=[("selected", ACCENT_COLOR)])

        self.notebook = ttk.Notebook(self.root, style="Dark.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self._setup_tab   = tk.Frame(self.notebook, bg=BG_COLOR)
        self._replay_tab  = tk.Frame(self.notebook, bg=BG_COLOR)
        self._calib_tab   = tk.Frame(self.notebook, bg=BG_COLOR)
        self._trim_tab    = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self._setup_tab,  text="  Setup  ")
        self.notebook.add(self._replay_tab, text="  Replay  ")
        self.notebook.add(self._calib_tab,  text="  Calibrate  ")
        self.notebook.add(self._trim_tab,   text="  Trim  ")
        # Face ID tab
        self._faceid_frame = tk.Frame(self.notebook, bg=BG_COLOR)
        self.notebook.add(self._faceid_frame, text="  [ID] Face ID  ")
        self._face_id_tab = FaceIDTab(self._faceid_frame, manager=self.face_mgr, app_root=APP_ROOT)

        self._build_setup_tab()
        self._build_replay_tab()
        self._build_calib_tab()
        self._build_trim_tab()
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

        self._status_var = tk.StringVar(value="Open a video to begin.")
        tk.Label(self.root, textvariable=self._status_var,
                 fg=ACCENT_COLOR, bg="#050505", anchor=tk.W,
                 font=("Consolas", 9)).pack(side=tk.BOTTOM, fill=tk.X, padx=8)

        self._prog_var = tk.DoubleVar()
        self._prog_bar = ttk.Progressbar(self.root, variable=self._prog_var,
                                         maximum=100)

        # Keyboard + mouse-wheel frame navigation (all tabs)
        self._bind_frame_nav()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            if self._face_id_tab:
                self._face_id_tab.stop_camera()
        except Exception:
            pass
        self.root.destroy()


    # -------------------------------------------------------------------------
    # Frame Navigation (keyboard a/d + mouse wheel) — active on all tabs
    # -------------------------------------------------------------------------

    def _bind_frame_nav(self):
        self.root.bind("<KeyPress-a>", self._on_key_prev)
        self.root.bind("<KeyPress-A>", self._on_key_prev)
        self.root.bind("<KeyPress-d>", self._on_key_next)
        self.root.bind("<KeyPress-D>", self._on_key_next)
        self.root.bind("<MouseWheel>", self._on_wheel)

    def _bind_canvas_nav(self, canvas):
        canvas.bind("<MouseWheel>", self._on_wheel)

    def _on_key_prev(self, _event=None):
        if self.state.frames:
            self._frame_step(-1)

    def _on_key_next(self, _event=None):
        if self.state.frames:
            self._frame_step(1)

    def _on_wheel(self, event):
        if not self.state.frames:
            return
        self._frame_step(-1 if event.delta < 0 else 1)

    def _frame_step(self, delta: int):
        if not self.state.frames:
            return
        total = len(self.state.frames)
        new_idx = max(0, min(self.state.current_frame + delta, total - 1))
        if new_idx == self.state.current_frame:
            return
        if self.state.is_playing and delta != 0:
            self.state.is_playing = False
            if hasattr(self, "_play_btn"):
                self._play_btn.config(text="Play")
        active = self.notebook.index(self.notebook.select())
        if active == 0:
            self._show_setup_frame(new_idx)
        elif active == 1:
            self._show_replay_frame(new_idx)
        elif active == 2:
            self.state.current_frame = new_idx
            self._refresh_calib()

    # -------------------------------------------------------------------------
    # Setup Tab
    # -------------------------------------------------------------------------

    def _build_setup_tab(self):
        # Face-ID name bar
        _id_bar = tk.Frame(self._setup_tab, bg="#0d1117")
        _id_bar.pack(fill=tk.X, side=tk.TOP)
        make_name_label(_id_bar).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(_id_bar, text="[ID] Identify Athlete",
                  command=self._open_identify_popup,
                  bg="#21262d", fg="#e0e0e0", relief=tk.FLAT,
                  font=("Segoe UI", 8), padx=10, pady=3).pack(side=tk.RIGHT, padx=4)
        self._setup_canvas = tk.Canvas(self._setup_tab, bg="#1a1a1a",
                                       cursor="crosshair",
                                       highlightthickness=1,
                                       highlightbackground=ACCENT_COLOR)
        self._setup_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                padx=(6, 2), pady=6)
        self._setup_canvas.bind("<Button-1>",  self._setup_left_click)
        self._setup_canvas.bind("<Button-3>",  self._setup_right_click)
        self._setup_canvas.bind("<Configure>", lambda e: self._refresh_setup())
        self._setup_canvas.bind("<Motion>",    lambda e: self._on_mag_motion(e, "setup"))
        self._setup_canvas.bind("<Leave>",     lambda e: self._on_mag_leave("setup"))
        self._bind_canvas_nav(self._setup_canvas)

        right = tk.Frame(self._setup_tab, bg=PANEL_BG, width=SIDEBAR_W)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 6), pady=6)
        right.pack_propagate(False)

        self._build_point_panel(right)
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        self._build_line_panel(right)

    def _build_point_panel(self, parent):
        hdr = tk.Frame(parent, bg=PANEL_BG)
        hdr.pack(fill=tk.X, padx=8, pady=(8, 2))
        tk.Label(hdr, text="TRACKED POINTS", fg=ACCENT_COLOR, bg=PANEL_BG,
                 font=("Consolas", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(hdr, text=f"(max {MAX_POINTS})", fg="#666", bg=PANEL_BG,
                 font=("Consolas", 8)).pack(side=tk.LEFT, padx=4)
        self._setup_mag_btn = tk.Button(
            hdr, text="Magnify", command=self._toggle_magnifier,
            bg=BTN_COLOR, fg=TEXT_COLOR, relief=tk.FLAT,
            padx=6, pady=2, font=("Segoe UI", 8), cursor="hand2")
        self._setup_mag_btn.pack(side=tk.RIGHT, padx=2)

        lb_frame = tk.Frame(parent, bg=PANEL_BG)
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        sb = tk.Scrollbar(lb_frame)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._pt_listbox = tk.Listbox(lb_frame, bg="#0d1117", fg=TEXT_COLOR,
                                      selectbackground="#1f6feb",
                                      font=("Consolas", 9), height=8,
                                      yscrollcommand=sb.set, relief=tk.FLAT,
                                      selectmode=tk.SINGLE)
        self._pt_listbox.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._pt_listbox.yview)
        self._pt_listbox.bind("<<ListboxSelect>>", self._on_pt_select)
        self._pt_listbox.bind("<Double-Button-1>",  lambda e: self._rename_point())

        ctrl = tk.Frame(parent, bg=PANEL_BG)
        ctrl.pack(fill=tk.X, padx=8, pady=2)
        _make_btn(ctrl, "Rename",    self._rename_point,    side=tk.LEFT, padx=2)
        _make_btn(ctrl, "Color",     self._change_pt_color, side=tk.LEFT, padx=2)
        _make_btn(ctrl, "Remove",    self._remove_point,    side=tk.LEFT, padx=2)
        _make_btn(ctrl, "Clear All", self._clear_all_points, side=tk.LEFT, padx=2)

        tk.Label(parent,
                 text="Click frame to place point  (auto-labelled P1, P2...)\n"
                      "Double-click list row or Rename to change label.\n"
                      "Right-click canvas to remove nearest point.",
                 fg="#666", bg=PANEL_BG, font=("Segoe UI", 8),
                 justify=tk.LEFT).pack(padx=8, pady=4, anchor=tk.W)

    def _build_line_panel(self, parent):
        tk.Label(parent, text="LINE CONNECTIONS", fg=ACCENT_COLOR, bg=PANEL_BG,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(4, 2))

        add_frame = tk.Frame(parent, bg=PANEL_BG)
        add_frame.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(add_frame, text="From:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._conn_a_var = tk.StringVar()
        self._conn_a_cb  = ttk.Combobox(add_frame, textvariable=self._conn_a_var,
                                        width=7, state="readonly",
                                        font=("Segoe UI", 8))
        self._conn_a_cb.pack(side=tk.LEFT, padx=2)
        tk.Label(add_frame, text="To:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 0))
        self._conn_b_var = tk.StringVar()
        self._conn_b_cb  = ttk.Combobox(add_frame, textvariable=self._conn_b_var,
                                        width=7, state="readonly",
                                        font=("Segoe UI", 8))
        self._conn_b_cb.pack(side=tk.LEFT, padx=2)

        add2 = tk.Frame(parent, bg=PANEL_BG)
        add2.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(add2, text="Label:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._conn_label_var = tk.StringVar()
        tk.Entry(add2, textvariable=self._conn_label_var, width=8,
                 bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                 font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

        self._conn_color_btn = tk.Button(add2, text="  Color  ",
                                         bg="#ffffff", fg="#000",
                                         relief=tk.FLAT, font=("Segoe UI", 8),
                                         cursor="hand2",
                                         command=self._pick_conn_color)
        self._conn_color_btn.pack(side=tk.LEFT, padx=2)
        self._conn_color  = (255, 255, 255)
        self._conn_dashed = False

        self._dash_btn = tk.Button(add2, text="Solid", bg="#1a1a2e", fg=TEXT_COLOR,
                                   relief=tk.FLAT, font=("Segoe UI", 8), cursor="hand2",
                                   command=self._toggle_dash)
        self._dash_btn.pack(side=tk.LEFT, padx=2)

        _make_btn(parent, "+ Add Connection", self._add_connection,
                  bg=ACCENT_COLOR, fg="#000", side=tk.TOP, padx=8)

        lb_frame2 = tk.Frame(parent, bg=PANEL_BG)
        lb_frame2.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        sb2 = tk.Scrollbar(lb_frame2)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self._conn_listbox = tk.Listbox(lb_frame2, bg="#0d1117", fg=TEXT_COLOR,
                                        selectbackground="#1f6feb",
                                        font=("Consolas", 8), height=5,
                                        yscrollcommand=sb2.set, relief=tk.FLAT)
        self._conn_listbox.pack(fill=tk.BOTH, expand=True)
        sb2.config(command=self._conn_listbox.yview)

        ctrl2 = tk.Frame(parent, bg=PANEL_BG)
        ctrl2.pack(fill=tk.X, padx=8, pady=2)
        _make_btn(ctrl2, "Remove Selected", self._remove_connection, side=tk.LEFT, padx=2)
        _make_btn(ctrl2, "Clear All",       self._clear_connections, side=tk.LEFT, padx=2)

    # -------------------------------------------------------------------------
    # Replay Tab
    # -------------------------------------------------------------------------

    def _build_replay_tab(self):
        # Face-ID name bar
        _id_bar2 = tk.Frame(self._replay_tab, bg="#0d1117")
        _id_bar2.pack(fill=tk.X, side=tk.TOP)
        make_name_label(_id_bar2).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(_id_bar2, text="[ID] Identify Athlete",
                  command=self._open_identify_popup,
                  bg="#21262d", fg="#e0e0e0", relief=tk.FLAT,
                  font=("Segoe UI", 8), padx=10, pady=3).pack(side=tk.RIGHT, padx=4)
        # Toolbar built first — _set_tool called at END after canvas exists
        self._build_annotation_toolbar()

        self._replay_canvas = tk.Canvas(self._replay_tab, bg="#1a1a1a",
                                        highlightthickness=1,
                                        highlightbackground=ACCENT_COLOR)
        self._replay_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self._replay_canvas.bind("<Button-1>",        self._annot_press)
        self._replay_canvas.bind("<B1-Motion>",       self._annot_drag)
        self._replay_canvas.bind("<ButtonRelease-1>", self._annot_release)
        self._replay_canvas.bind("<Configure>",       lambda e: self._refresh_replay())
        self._bind_canvas_nav(self._replay_canvas)

        # ── Bottom controls: smoothing + kinematics side-by-side ─────────────
        bottom_row = tk.Frame(self._replay_tab, bg="#0d1117")
        bottom_row.pack(fill=tk.X, padx=6, pady=(0, 0))

        # Left: smoothing
        smooth_bar = tk.Frame(bottom_row, bg=PANEL_BG, pady=3)
        smooth_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(smooth_bar, text="Smooth:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(6, 2))
        self._smooth_var = tk.BooleanVar(value=SMOOTH_DEFAULT_ENABLED)
        tk.Checkbutton(smooth_bar, text="On", variable=self._smooth_var,
                       bg=PANEL_BG, fg=TEXT_COLOR, selectcolor=PANEL_BG,
                       activebackground=PANEL_BG, activeforeground=ACCENT_COLOR,
                       font=("Segoe UI", 8),
                       command=self._on_smooth_change).pack(side=tk.LEFT, padx=2)
        tk.Label(smooth_bar, text="Win:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._smooth_win_var = tk.IntVar(value=SMOOTH_WINDOW_DEFAULT)
        tk.Spinbox(smooth_bar, from_=3, to=99, increment=2,
                   textvariable=self._smooth_win_var, width=3,
                   bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                   font=("Segoe UI", 8),
                   command=self._on_smooth_change).pack(side=tk.LEFT)
        tk.Label(smooth_bar, text="Poly:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 1))
        self._smooth_poly_var = tk.IntVar(value=SMOOTH_POLYORDER_DEFAULT)
        tk.Spinbox(smooth_bar, from_=1, to=9, increment=1,
                   textvariable=self._smooth_poly_var, width=2,
                   bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                   font=("Segoe UI", 8),
                   command=self._on_smooth_change).pack(side=tk.LEFT)
        _make_btn(smooth_bar, "Apply", self._on_smooth_change,
                  bg=ACCENT_COLOR, fg="#000", side=tk.LEFT, padx=4)
        ttk.Separator(smooth_bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, pady=3, padx=4)
        self._trails_var = tk.BooleanVar(value=True)
        tk.Checkbutton(smooth_bar, text="Trails", variable=self._trails_var,
                       bg=PANEL_BG, fg=TEXT_COLOR, selectcolor=PANEL_BG,
                       activebackground=PANEL_BG, font=("Segoe UI", 8),
                       command=self._refresh_replay).pack(side=tk.LEFT, padx=2)
        self._labels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(smooth_bar, text="Labels", variable=self._labels_var,
                       bg=PANEL_BG, fg=TEXT_COLOR, selectcolor=PANEL_BG,
                       activebackground=PANEL_BG, font=("Segoe UI", 8),
                       command=self._refresh_replay).pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(bottom_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, pady=2, padx=2)

        # Right: kinematics
        kin_bar = tk.Frame(bottom_row, bg="#0d1117", pady=3)
        kin_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(kin_bar, text="Kinematics:", fg=TEXT_COLOR, bg="#0d1117",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(6, 2))
        _make_btn(kin_bar, "Compute & Export",
                  self._compute_export_kinematics,
                  bg="#1f6feb", fg=TEXT_COLOR, side=tk.LEFT, padx=3)
        tk.Label(kin_bar, text="Win:", fg=TEXT_COLOR, bg="#0d1117",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(6, 1))
        self._kin_win_var = tk.IntVar(value=11)
        tk.Spinbox(kin_bar, from_=3, to=99, increment=2,
                   textvariable=self._kin_win_var, width=3,
                   bg="#0d1117", fg=TEXT_COLOR, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        tk.Label(kin_bar, text="Poly:", fg=TEXT_COLOR, bg="#0d1117",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 1))
        self._kin_poly_var = tk.IntVar(value=3)
        tk.Spinbox(kin_bar, from_=1, to=9, increment=1,
                   textvariable=self._kin_poly_var, width=2,
                   bg="#0d1117", fg=TEXT_COLOR, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._kin_status_lbl = tk.Label(
            kin_bar, text="Calibrate tab first.",
            fg="#555", bg="#0d1117", font=("Segoe UI", 7))
        self._kin_status_lbl.pack(side=tk.LEFT, padx=6)


        # Playback bar (full width, below smooth+kin row)
        play_bar = tk.Frame(self._replay_tab, bg="#111111", pady=3)
        play_bar.pack(fill=tk.X, padx=6)

        self._play_btn = _make_btn(play_bar, "Play", self._toggle_play,
                                   side=tk.LEFT, padx=4)
        _make_btn(play_bar, "|<", lambda: self._seek(0),
                  side=tk.LEFT, padx=2)
        _make_btn(play_bar, "<<", lambda: self._seek(self.state.current_frame - 10),
                  side=tk.LEFT, padx=2)
        _make_btn(play_bar, ">>", lambda: self._seek(self.state.current_frame + 10),
                  side=tk.LEFT, padx=2)
        _make_btn(play_bar, ">|", lambda: self._seek(len(self.state.frames) - 1),
                  side=tk.LEFT, padx=2)

        self._replay_scrub = ttk.Scale(play_bar, from_=0, to=1,
                                       orient=tk.HORIZONTAL,
                                       command=self._on_replay_scrub)
        self._replay_scrub.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        tk.Label(play_bar, text="Speed:", fg=TEXT_COLOR, bg="#111111",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._speed_var = tk.DoubleVar(value=1.0)
        tk.Spinbox(play_bar, from_=0.25, to=4.0, increment=0.25,
                   textvariable=self._speed_var, width=4,
                   bg="#0d1117", fg=TEXT_COLOR,
                   font=("Segoe UI", 8)).pack(side=tk.LEFT)

        self._replay_lbl = tk.Label(play_bar, text="Frame: 0 / 0",
                                    fg=TEXT_COLOR, bg="#111111",
                                    font=("Consolas", 9))
        self._replay_lbl.pack(side=tk.RIGHT, padx=8)
        # _replay_canvas now exists — safe to set initial tool
        self._set_tool("select")

    def _build_annotation_toolbar(self):
        bar = tk.Frame(self._replay_tab, bg="#0d1117", pady=5)
        bar.pack(fill=tk.X, padx=6, pady=(6, 0))

        tk.Label(bar, text="Annotations:", fg="#888", bg="#0d1117",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 6))

        self._tool_btns = {}
        tools = [
            ("Select",  "select"),
            ("Line",    "line"),
            ("Arrow",   "arrow"),
            ("Rect",    "rect"),
            ("Ellipse", "ellipse"),
            ("Text",    "text"),
            ("Angle",   "angle"),
        ]
        for label, key in tools:
            btn = tk.Button(bar, text=label, bg=BTN_COLOR, fg=TEXT_COLOR,
                            relief=tk.FLAT, padx=8, pady=3,
                            font=("Segoe UI", 8), cursor="hand2",
                            command=lambda k=key: self._set_tool(k))
            btn.pack(side=tk.LEFT, padx=2)
            self._tool_btns[key] = btn

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y,
                                                    padx=8, pady=2)

        self._annot_color_btn = tk.Button(
            bar, text="  Color  ", bg=self._annot_color, fg="#000",
            relief=tk.FLAT, font=("Segoe UI", 8), cursor="hand2",
            command=self._pick_annot_color)
        self._annot_color_btn.pack(side=tk.LEFT, padx=4)

        tk.Label(bar, text="Width:", fg=TEXT_COLOR, bg="#0d1117",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._annot_width_var = tk.IntVar(value=config.ANNOT_DEFAULT_WIDTH)
        tk.Spinbox(bar, from_=1, to=10, textvariable=self._annot_width_var,
                   width=3, bg="#0d1117", fg=TEXT_COLOR,
                   font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=2)

        self._global_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="All frames", variable=self._global_var,
                       bg="#0d1117", fg=TEXT_COLOR, selectcolor="#0d1117",
                       font=("Segoe UI", 8),
                       activebackground="#0d1117").pack(side=tk.LEFT, padx=8)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        _make_btn(bar, "Undo",        self._annot_undo,        side=tk.LEFT, padx=2)
        _make_btn(bar, "Clear Frame", self._annot_clear_frame, side=tk.LEFT, padx=2)
        _make_btn(bar, "Clear All",   self._annot_clear_all,   side=tk.LEFT, padx=2)
        # _set_tool NOT called here -- _replay_canvas not yet created at this point

    # =========================================================================
    # Video Loading
    # =========================================================================

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Open Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4"),
                       ("All", "*.*")]
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        try:
            self.state.video           = VideoReader(path)
            self.state.current_frame   = 0
            self.state.tracking_done   = False
            self.state.result          = None
            self.state.smoothed_tracks = None
            self.state.frames          = []
            self.state.query_points    = []
            self.state.point_labels    = []
            self.state.point_colors    = []
            self.state.connections     = []
            self.state.annot_manager.clear_all()
            self.state.calib_mgr.clear()
            self._pt_listbox.delete(0, tk.END)
            self._conn_listbox.delete(0, tk.END)
            if hasattr(self, "_lin_listbox"):
                self._lin_listbox.delete(0, tk.END)
            if hasattr(self, "_fld_listbox"):
                self._fld_listbox.delete(0, tk.END)
            if hasattr(self, "_assign_listbox"):
                self._assign_listbox.delete(0, tk.END)
            self._calib_click_pts = []

            self._status(
                f"Loaded: {os.path.basename(path)}  "
                f"({self.state.video.frame_count} frames  "
                f"{self.state.video.fps:.1f} fps  "
                f"{self.state.video.orig_width}x{self.state.video.orig_height}  "
                f"display: {self.state.video.width}x{self.state.video.height})"
            )
            threading.Thread(target=self._read_frames_bg, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _read_frames_bg(self):
        self._prog_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        try:
            self.state.frames = self.state.video.read_all_frames(
                callback=lambda i, total: self._prog_var.set(
                    100 * i / max(total, 1))
            )
        except (MemoryError, cv2.error, Exception) as e:
            self._prog_bar.pack_forget()
            self._status("Error: out of memory reading frames.")
            self.root.after(0, lambda: messagebox.showerror(
                "Out of Memory",
                f"Failed to load video frames.\n\n"
                f"Try lowering DISPLAY_MAX_W / DISPLAY_MAX_H in config.py\n"
                f"(currently {config.DISPLAY_MAX_W}x{config.DISPLAY_MAX_H}).\n\n"
                f"Detail: {e}"
            ))
            return
        self._prog_bar.pack_forget()
        self._replay_scrub.config(to=max(len(self.state.frames) - 1, 1))
        self._status(
            f"Ready -- {len(self.state.frames)} frames loaded  "
            f"({self.state.video.width}x{self.state.video.height} display res).  "
            f"Click frame to place tracked points (max {MAX_POINTS}).")
        self.root.after(0, lambda: self._show_setup_frame(0))

    # =========================================================================
    # Setup Tab Logic
    # =========================================================================

    def _refresh_setup(self):
        self._show_setup_frame(self.state.current_frame)

    def _show_setup_frame(self, idx: int):
        if not self.state.frames:
            return
        idx   = max(0, min(idx, len(self.state.frames) - 1))
        self.state.current_frame = idx
        frame = self.state.frames[idx].copy()

        for i, (xn, yn) in enumerate(self.state.query_points):
            h, w   = frame.shape[:2]
            cx, cy = int(xn * w), int(yn * h)
            color  = (self.state.point_colors[i]
                      if i < len(self.state.point_colors)
                      else COLORS[i % len(COLORS)])
            _r = max(1, config.POINT_RADIUS)
            _dot = frame.copy()
            cv2.circle(_dot, (cx, cy), _r,     color, -1, cv2.LINE_AA)
            cv2.circle(_dot, (cx, cy), _r + 2, color,  1, cv2.LINE_AA)
            cv2.addWeighted(_dot, 0.5, frame, 0.5, 0, frame)
            tag = (self.state.point_labels[i]
                   if i < len(self.state.point_labels) else f"P{i+1}")
            cv2.putText(frame, tag, (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        self._blit_to_canvas(self._setup_canvas, frame, self.state, "setup")

    def _setup_left_click(self, event):
        if not self.state.frames:
            return
        if len(self.state.query_points) >= MAX_POINTS:
            self._status(f"Max {MAX_POINTS} points reached. Right-click to remove.")
            return
        if self.state.tracking_done:
            self._status("Clear tracking first to reposition points.")
            return

        xn, yn = self._canvas_to_norm(event.x, event.y, self.state, "setup")

        existing = set(self.state.point_labels)
        n = len(self.state.query_points) + 1
        while f"P{n}" in existing:
            n += 1
        label = f"P{n}"
        color = COLORS[len(self.state.query_points) % len(COLORS)]

        self.state.query_points.append((xn, yn))
        self.state.point_labels.append(label)
        self.state.point_colors.append(color)
        self.state.query_frame = self.state.current_frame

        self._pt_listbox.insert(tk.END, f"  {label}")
        self._pt_listbox.itemconfig(tk.END, fg=bgr_to_hex(color))
        self._update_conn_combos()
        if hasattr(self, '_assign_pt_cb'):
            self._update_assign_combos()
        self._status(
            f"Point '{label}' placed -- [{len(self.state.query_points)}/{MAX_POINTS}]  "
            "Double-click list row or use Rename to change label."
        )
        self._show_setup_frame(self.state.current_frame)

    def _setup_right_click(self, event):
        if not self.state.query_points:
            return
        xn, yn = self._canvas_to_norm(event.x, event.y, self.state, "setup")
        dists   = [((xn - p[0]) ** 2 + (yn - p[1]) ** 2) ** 0.5
                   for p in self.state.query_points]
        i       = int(np.argmin(dists))
        removed = self.state.point_labels[i]
        self.state.query_points.pop(i)
        self.state.point_labels.pop(i)
        self.state.point_colors.pop(i)
        self._pt_listbox.delete(i)
        self._update_conn_combos()
        if hasattr(self, '_assign_pt_cb'):
            self._update_assign_combos()
        self._status(f"Removed point '{removed}'.")
        self._show_setup_frame(self.state.current_frame)

    # -- Point panel actions --------------------------------------------------

    def _on_pt_select(self, _event=None):
        pass

    def _rename_point(self):
        sel = self._pt_listbox.curselection()
        if not sel:
            return
        i   = sel[0]
        old = self.state.point_labels[i]
        new = self._text_dialog("Rename Point", old, "New label:")
        if new and new != old:
            self.state.point_labels[i] = new
            color = self.state.point_colors[i]
            self._pt_listbox.delete(i)
            self._pt_listbox.insert(i, f"  {new}")
            self._pt_listbox.itemconfig(i, fg=bgr_to_hex(color))
            self._update_conn_combos()
            if hasattr(self, '_assign_pt_cb'):
                self._update_assign_combos()
            self._show_setup_frame(self.state.current_frame)

    def _change_pt_color(self):
        sel = self._pt_listbox.curselection()
        if not sel:
            return
        i      = sel[0]
        chosen = colorchooser.askcolor(
            color=bgr_to_hex(self.state.point_colors[i]),
            title="Choose Point Color"
        )
        if chosen and chosen[1]:
            r, g, b = int(chosen[0][0]), int(chosen[0][1]), int(chosen[0][2])
            self.state.point_colors[i] = (b, g, r)
            self._pt_listbox.itemconfig(i, fg=chosen[1])
            self._show_setup_frame(self.state.current_frame)

    def _remove_point(self):
        sel = self._pt_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        self.state.query_points.pop(i)
        self.state.point_labels.pop(i)
        self.state.point_colors.pop(i)
        self._pt_listbox.delete(i)
        self._update_conn_combos()
        if hasattr(self, '_assign_pt_cb'):
            self._update_assign_combos()
        self._show_setup_frame(self.state.current_frame)

    def _clear_all_points(self):
        self.state.query_points    = []
        self.state.point_labels    = []
        self.state.point_colors    = []
        self.state.tracking_done   = False
        self.state.result          = None
        self.state.smoothed_tracks = None
        self._pt_listbox.delete(0, tk.END)
        self._update_conn_combos()
        self._status("All points cleared.")
        self._show_setup_frame(self.state.current_frame)

    # -- Line connection actions -----------------------------------------------

    def _update_conn_combos(self):
        labels = self.state.point_labels
        self._conn_a_cb["values"] = labels
        self._conn_b_cb["values"] = labels

    def _pick_conn_color(self):
        chosen = colorchooser.askcolor(
            color=bgr_to_hex(self._conn_color), title="Connection Color")
        if chosen and chosen[1]:
            r, g, b = int(chosen[0][0]), int(chosen[0][1]), int(chosen[0][2])
            self._conn_color = (b, g, r)
            self._conn_color_btn.config(bg=chosen[1])

    def _toggle_dash(self):
        self._conn_dashed = not self._conn_dashed
        self._dash_btn.config(text="Dashed" if self._conn_dashed else "Solid")

    def _add_connection(self):
        la = self._conn_a_var.get()
        lb = self._conn_b_var.get()
        if not la or not lb:
            messagebox.showwarning("Select Points", "Choose both From and To points.")
            return
        if la == lb:
            messagebox.showwarning("Same Point", "From and To must be different points.")
            return
        labels = self.state.point_labels
        if la not in labels or lb not in labels:
            messagebox.showwarning("Unknown Point", "Point not found in tracking list.")
            return
        ia, ib = labels.index(la), labels.index(lb)

        for existing in self.state.connections:
            if existing.idx_a == ia and existing.idx_b == ib:
                if not messagebox.askyesno(
                    "Duplicate Connection",
                    f"A connection from '{la}' to '{lb}' already exists.\n"
                    "Add a second one anyway?"
                ):
                    return
                break

        lbl  = self._conn_label_var.get().strip()
        conn = LineConnection(idx_a=ia, idx_b=ib,
                              color_bgr=self._conn_color,
                              thickness=2, label=lbl,
                              dashed=self._conn_dashed)
        self.state.connections.append(conn)
        sep  = "- -" if self._conn_dashed else "---"
        disp = f"{la} {sep} {lb}"
        if lbl:
            disp += f"  [{lbl}]"
        self._conn_listbox.insert(tk.END, disp)
        self._conn_listbox.itemconfig(tk.END, fg=bgr_to_hex(self._conn_color))
        n_conns = sum(1 for c in self.state.connections
                      if labels[c.idx_a] == la or labels[c.idx_b] == la)
        self._status(
            f"Connection added: {la} -> {lb}  "
            f"('{la}' is now in {n_conns} connection(s))"
        )
        self._conn_b_var.set("")

    def _remove_connection(self):
        sel = self._conn_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        self.state.connections.pop(i)
        self._conn_listbox.delete(i)

    def _clear_connections(self):
        self.state.connections = []
        self._conn_listbox.delete(0, tk.END)

    # =========================================================================
    # Tracking Inference
    # =========================================================================

    def _run_tracking(self):
        if not self.state.frames:
            messagebox.showwarning("No Video", "Please open a video first.")
            return
        if not self.state.query_points:
            messagebox.showwarning("No Points",
                                   "Place at least one point on the Setup frame.")
            return
        self._status("Loading model...")
        self._prog_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        self._prog_var.set(0)
        threading.Thread(target=self._tracking_thread, daemon=True).start()

    def _tracking_thread(self):
        try:
            backend = self._engine_var.get().lower()
            # Release previous engine GPU memory before loading new one
            engine  = build_engine(backend,
                                   previous_engine=self.state.engine)
            self.state.engine = engine
            engine_name = type(engine).__name__.replace("Engine", "")
            self._status(f"Running {engine_name} inference...")

            frames_rgb = frames_to_rgb_tensor(self.state.frames)
            T, H, W, _ = frames_rgb.shape
            qf         = float(self.state.query_frame)  # TAPIR needs absolute frame index
            query_pts  = np.array(
                [[qf, yn, xn] for xn, yn in self.state.query_points],
                dtype=np.float32
            )

            result = engine.track(frames_rgb, query_pts, orig_hw=(H, W))
            self.state.result        = result
            self.state.tracking_done = True

            self._status("Applying smoothing...")
            self._apply_smoothing(silent=True)

            self._prog_bar.pack_forget()
            self._status("Tracking complete -- switch to Replay tab.")
            self.root.after(0, lambda: self.notebook.select(self._replay_tab))
            self.root.after(50, lambda: self._show_replay_frame(0))

        except Exception as e:
            self._prog_bar.pack_forget()
            err_msg = str(e)
            self._status(f"Error: {err_msg}")
            logger.exception("Tracking failed")
            self.root.after(0,
                lambda m=err_msg: messagebox.showerror("Tracking Error", m))

    # =========================================================================
    # Smoothing
    # =========================================================================

    def _on_smooth_change(self, *_):
        if not self.state.tracking_done:
            return
        self._apply_smoothing()

    def _apply_smoothing(self, silent: bool = False):
        if self.state.result is None:
            return
        r      = self.state.result
        window = (self._smooth_win_var.get()
                  if hasattr(self, "_smooth_win_var") else SMOOTH_WINDOW_DEFAULT)
        poly   = (self._smooth_poly_var.get()
                  if hasattr(self, "_smooth_poly_var") else SMOOTH_POLYORDER_DEFAULT)
        smooth = (self._smooth_var.get()
                  if hasattr(self, "_smooth_var") else SMOOTH_DEFAULT_ENABLED)

        self.state.smoothed_tracks = process_tracks(
            r.tracks, r.occlusions,
            interp=True, smooth=smooth,
            window_length=window, polyorder=poly,
        )
        if not silent:
            self._status(f"Smoothing applied (window={window}, poly={poly}).")
            self._show_replay_frame(self.state.current_frame)

    # =========================================================================
    # Replay Tab Logic
    # =========================================================================

    def _refresh_replay(self):
        self._show_replay_frame(self.state.current_frame)

    def _show_replay_frame(self, idx: int):
        if not self.state.frames:
            return
        idx = max(0, min(idx, len(self.state.frames) - 1))
        self.state.current_frame = idx
        frame = self.state.frames[idx].copy()

        if self.state.tracking_done and self.state.result is not None:
            tracks = (self.state.smoothed_tracks
                      if self.state.smoothed_tracks is not None
                      else self.state.result.tracks)
            frame = draw_tracks(
                frame, idx, tracks,
                self.state.result.occlusions,
                self.state.result.scores,
                labels      = self.state.point_labels,
                connections = self.state.connections,
                show_trails = self._trails_var.get(),
                show_labels = self._labels_var.get(),
            )
            frame = draw_hud(
                frame, idx, len(self.state.frames),
                self.state.video.fps,
                len(self.state.query_points),
                smoothed=(self.state.smoothed_tracks is not None
                          and self._smooth_var.get()),
            )

        annots = self.state.annot_manager.get(idx)
        if annots:
            frame = draw_annotations(frame, annots)

        self._blit_to_canvas(self._replay_canvas, frame, self.state, "replay")
        total = len(self.state.frames)
        self._replay_lbl.config(text=f"Frame: {idx+1} / {total}")
        self._scrub_updating = True
        try:
            self._replay_scrub.set(idx)
        finally:
            self._scrub_updating = False

    # -- Display helper --------------------------------------------------------

    def _blit_to_canvas(self, canvas: tk.Canvas, frame: np.ndarray,
                        state: AppState, mode: str):
        # Use actual canvas size; guard against unrealised canvas (reports 1px)
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 50: cw = CANVAS_MAX_W
        if ch < 50: ch = CANVAS_MAX_H
        preview = frames_to_preview(frame, cw, ch)
        img_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        tk_img  = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        ox = max(0, (cw - preview.shape[1]) // 2)
        oy = max(0, (ch - preview.shape[0]) // 2)

        canvas.delete("all")
        canvas.create_image(ox, oy, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img

        dh, dw = preview.shape[:2]
        oh, ow = frame.shape[:2]
        scale  = dw / ow
        if mode == "setup":
            state.setup_scale = scale; state.setup_ox = ox; state.setup_oy = oy
            state.setup_fw = ow;       state.setup_fh = oh
        elif mode == "calib":
            state.calib_scale = scale; state.calib_ox = ox; state.calib_oy = oy
            state.calib_fw = ow;       state.calib_fh = oh
        else:
            state.replay_scale = scale; state.replay_ox = ox; state.replay_oy = oy
            state.replay_fw = ow;       state.replay_fh = oh

    def _canvas_to_norm(self, cx: int, cy: int, state: AppState, mode: str):
        if mode == "setup":
            scale, ox, oy, fw, fh = (state.setup_scale, state.setup_ox,
                                     state.setup_oy, state.setup_fw, state.setup_fh)
        elif mode == "calib":
            scale, ox, oy, fw, fh = (state.calib_scale, state.calib_ox,
                                     state.calib_oy, state.calib_fw, state.calib_fh)
        else:
            scale, ox, oy, fw, fh = (state.replay_scale, state.replay_ox,
                                     state.replay_oy, state.replay_fw, state.replay_fh)
        xp = max(0, min((cx - ox) / scale, fw - 1)) / fw
        yp = max(0, min((cy - oy) / scale, fh - 1)) / fh
        return xp, yp

    # =========================================================================
    # Annotation Tools
    # =========================================================================

    def _set_tool(self, key: str):
        self._annot_tool = key
        self._replay_canvas.config(
            cursor="crosshair" if key != "select" else "arrow")
        for k, btn in self._tool_btns.items():
            btn.config(bg=ACCENT_COLOR if k == key else BTN_COLOR,
                       fg="#000"       if k == key else TEXT_COLOR)
        self._draw_start     = None
        self._draw_angle_pts = []
        if self._temp_line_id:
            self._replay_canvas.delete(self._temp_line_id)
            self._temp_line_id = None

    def _pick_annot_color(self):
        chosen = colorchooser.askcolor(color=self._annot_color,
                                       title="Annotation Color")
        if chosen and chosen[1]:
            self._annot_color = chosen[1]
            self._annot_color_btn.config(bg=self._annot_color)

    def _annot_press(self, event):
        tool = self._annot_tool
        if tool == "select":
            return
        xn, yn = self._canvas_to_norm(event.x, event.y, self.state, "replay")

        if tool == "angle":
            self._draw_angle_pts.append((xn, yn))
            if len(self._draw_angle_pts) == 3:
                p = self._draw_angle_pts
                color_bgr = hex_to_bgr(self._annot_color)
                obj = AnnotAngle(p[0][0], p[0][1], p[1][0], p[1][1],
                                 p[2][0], p[2][1],
                                 color=color_bgr,
                                 width=self._annot_width_var.get())
                self.state.annot_manager.add(
                    self.state.current_frame, obj,
                    global_annot=self._global_var.get())
                self._draw_angle_pts = []
                self._show_replay_frame(self.state.current_frame)
            return

        if tool == "text":
            text = self._text_dialog("Add Text", "", "Annotation text:")
            if text:
                color_bgr = hex_to_bgr(self._annot_color)
                obj = AnnotText(xn, yn, text, color=color_bgr,
                                size=config.ANNOT_DEFAULT_FONT_SIZE / 26.0)
                self.state.annot_manager.add(
                    self.state.current_frame, obj,
                    global_annot=self._global_var.get())
                self._show_replay_frame(self.state.current_frame)
            return

        self._draw_start = (xn, yn)

    def _annot_drag(self, event):
        if self._annot_tool in ("select", "text", "angle") or not self._draw_start:
            return
        if self._temp_line_id:
            self._replay_canvas.delete(self._temp_line_id)
        x0, y0 = self._draw_start
        cx0 = int(x0 * self.state.replay_fw * self.state.replay_scale
                  + self.state.replay_ox)
        cy0 = int(y0 * self.state.replay_fh * self.state.replay_scale
                  + self.state.replay_oy)
        self._temp_line_id = self._replay_canvas.create_line(
            cx0, cy0, event.x, event.y,
            fill=self._annot_color, width=self._annot_width_var.get(),
            dash=(4, 4) if self._annot_tool == "rect" else None,
        )

    def _annot_release(self, event):
        tool = self._annot_tool
        if tool in ("select", "text", "angle") or not self._draw_start:
            return
        if self._temp_line_id:
            self._replay_canvas.delete(self._temp_line_id)
            self._temp_line_id = None

        x0, y0    = self._draw_start
        xn, yn    = self._canvas_to_norm(event.x, event.y, self.state, "replay")
        color_bgr = hex_to_bgr(self._annot_color)
        width     = self._annot_width_var.get()
        glb       = self._global_var.get()
        obj       = None

        if tool == "line":
            obj = AnnotLine(x0, y0, xn, yn, color=color_bgr, width=width)
        elif tool == "arrow":
            obj = AnnotArrow(x0, y0, xn, yn, color=color_bgr, width=width)
        elif tool == "rect":
            obj = AnnotRect(x0, y0, xn, yn, color=color_bgr, width=width)
        elif tool == "ellipse":
            cx = (x0 + xn) / 2;  cy = (y0 + yn) / 2
            rx = abs(xn - x0) / 2; ry = abs(yn - y0) / 2
            obj = AnnotEllipse(cx, cy, rx, ry, color=color_bgr, width=width)

        if obj:
            self.state.annot_manager.add(self.state.current_frame, obj,
                                         global_annot=glb)
            self._show_replay_frame(self.state.current_frame)
        self._draw_start = None

    def _annot_undo(self):
        self.state.annot_manager.pop_last(self.state.current_frame)
        self._show_replay_frame(self.state.current_frame)

    def _annot_clear_frame(self):
        self.state.annot_manager.clear_frame(self.state.current_frame)
        self._show_replay_frame(self.state.current_frame)

    def _annot_clear_all(self):
        if messagebox.askyesno("Clear Annotations",
                               "Clear all annotations on all frames?"):
            self.state.annot_manager.clear_all()
            self._show_replay_frame(self.state.current_frame)

    # =========================================================================
    # Playback
    # =========================================================================

    def _toggle_play(self):
        if not self.state.frames:
            return
        self.state.is_playing = not self.state.is_playing
        self._play_btn.config(
            text="Pause" if self.state.is_playing else "Play")
        if self.state.is_playing:
            self._play_loop()

    def _play_loop(self):
        if not self.state.is_playing:
            return
        fps   = self.state.video.fps if self.state.video else 25.0
        speed = self._speed_var.get() if hasattr(self, "_speed_var") else 1.0
        delay = max(1, int(1000 / (fps * speed)))
        idx   = self.state.current_frame + 1
        if idx >= len(self.state.frames):
            self.state.is_playing = False
            self._play_btn.config(text="Play")
            self._show_replay_frame(0)
            return
        self._show_replay_frame(idx)
        self._after_id = self.root.after(delay, self._play_loop)

    def _seek(self, idx: int):
        if self.state.is_playing:
            self.state.is_playing = False
            self._play_btn.config(text="Play")
        self._show_replay_frame(idx)

    def _on_replay_scrub(self, val):
        if getattr(self, "_scrub_updating", False):
            return   # programmatic set() call -- ignore to avoid recursion
        if self.state.is_playing:
            self.state.is_playing = False
            self._play_btn.config(text="Play")
        self._show_replay_frame(int(float(val)))

    # =========================================================================
    # Export
    # =========================================================================

    def _export(self):
        if not self.state.tracking_done or self.state.result is None:
            messagebox.showwarning("Nothing to Export", "Run tracking first.")
            return
        self._status("Exporting...")
        threading.Thread(target=self._export_thread, daemon=True).start()

    def _export_thread(self):
        try:
            r      = self.state.result
            tracks = (self.state.smoothed_tracks
                      if self.state.smoothed_tracks is not None else r.tracks)
            T      = len(self.state.frames)
            fps    = self.state.video.fps

            writer = VideoWriter(self.state.video)
            for i, frame in enumerate(self.state.frames):
                f = draw_tracks(frame, i, tracks, r.occlusions, r.scores,
                                labels=self.state.point_labels,
                                connections=self.state.connections,
                                show_trails=self._trails_var.get(),
                                show_labels=self._labels_var.get())
                f = draw_hud(f, i, T, fps, len(self.state.query_points),
                             smoothed=(self.state.smoothed_tracks is not None))
                annots = self.state.annot_manager.get(i)
                if annots:
                    f = draw_annotations(f, annots)
                writer.write(f)
            writer.close()

            csv_path = export_tracks_csv(
                tracks=tracks, occlusions=r.occlusions, scores=r.scores,
                fps=fps, video_name=self.state.video.path,
                labels=self.state.point_labels,
            )

            msg = f"Video:\n  {writer.out_path}"
            if csv_path:
                msg += f"\n\nCSV:\n  {csv_path}"
            self._status(f"Exported -> {writer.out_path}")
            self.root.after(0, lambda: messagebox.showinfo("Export Complete", msg))

        except Exception as e:
            err_msg = str(e)
            self._status(f"Export failed: {err_msg}")
            logger.exception("Export failed")
            self.root.after(0,
                lambda m=err_msg: messagebox.showerror("Export Error", m))


    # =========================================================================
    # Trim Tab
    # =========================================================================

    def _build_trim_tab(self):
        # Face-ID name bar
        _id_bar4 = tk.Frame(self._trim_tab, bg="#0d1117")
        _id_bar4.pack(fill=tk.X, side=tk.TOP)
        make_name_label(_id_bar4).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(_id_bar4, text="[ID] Identify Athlete",
                  command=self._open_identify_popup,
                  bg="#21262d", fg="#e0e0e0", relief=tk.FLAT,
                  font=("Segoe UI", 8), padx=10, pady=3).pack(side=tk.RIGHT, padx=4)
        """Build the video trim tab — matches tracker-rtm / tracker-pose UI."""
        bar = tk.Frame(self._trim_tab, bg=PANEL_BG, height=44)
        bar.pack(fill=tk.X); bar.pack_propagate(False)
        tk.Label(bar, text="Video Trim", bg=PANEL_BG, fg=ACCENT_COLOR,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=14, pady=12)
        tk.Button(bar, text="Load source", command=self._trim_load,
                  bg=ACCENT_COLOR, fg="#000", relief=tk.FLAT,
                  font=("Segoe UI", 8, "bold"), padx=10).pack(side=tk.LEFT, padx=4)
        self._trim_btn_play = tk.Button(
            bar, text="\u25b6  Play", command=self._trim_toggle_play,
            bg=BTN_COLOR, fg=TEXT_COLOR, relief=tk.FLAT,
            font=("Segoe UI", 8), padx=8, state=tk.DISABLED)
        self._trim_btn_play.pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Reset", command=self._trim_reset,
                  bg=BTN_COLOR, fg=TEXT_COLOR, relief=tk.FLAT,
                  font=("Segoe UI", 8), padx=8).pack(side=tk.LEFT, padx=4)
        self._trim_btn_save = tk.Button(
            bar, text="\U0001f4be  Save & Load", command=self._trim_save_and_load,
            bg="#1a5c1a", fg="#aaffaa", relief=tk.FLAT,
            font=("Segoe UI", 8, "bold"), padx=10, state=tk.DISABLED)
        self._trim_btn_save.pack(side=tk.LEFT, padx=4)
        cv_fr = tk.Frame(self._trim_tab, bg="#0a0a0a",
                         highlightthickness=1, highlightbackground="#2a2a3a")
        cv_fr.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 0))
        self._trim_canvas = tk.Canvas(cv_fr, bg="#0a0a0a", highlightthickness=0)
        self._trim_canvas.pack(fill=tk.BOTH, expand=True)
        self._trim_canvas.bind("<Configure>", lambda e: self._trim_blit())
        self._trim_hint = tk.Label(
            self._trim_canvas,
            text="Click  Load source  to begin  (or switch away then back to auto-load current video)",
            bg="#0a0a0a", fg="#555", font=("Segoe UI", 11))
        self._trim_hint.place(relx=0.5, rely=0.5, anchor="center")
        tl = tk.Frame(self._trim_tab, bg="#0d1117",
                      highlightthickness=1, highlightbackground="#333")
        tl.pack(fill=tk.X, padx=10, pady=(4, 0))
        ph = tk.Frame(tl, bg="#0d1117"); ph.pack(fill=tk.X, padx=8, pady=(4, 0))
        self._trim_lbl_pos = tk.Label(ph, text="00:00:00.00",
                                      bg="#0d1117", fg="#888", font=("Consolas", 8))
        self._trim_lbl_pos.pack(side=tk.LEFT)
        self._trim_lbl_src = tk.Label(ph, text="No source loaded",
                                      bg="#0d1117", fg="#555", font=("Segoe UI", 8))
        self._trim_lbl_src.pack(side=tk.RIGHT)
        self._trim_sc = tk.Canvas(tl, height=80, bg="#0d1117",
                                  highlightthickness=0, cursor="hand2")
        self._trim_sc.pack(fill=tk.X, padx=8, pady=(2, 4))
        self._trim_sc.bind("<Configure>",       lambda e: self._trim_draw_slider())
        self._trim_sc.bind("<ButtonPress-1>",   self._trim_press)
        self._trim_sc.bind("<B1-Motion>",       self._trim_drag_move)
        self._trim_sc.bind("<ButtonRelease-1>", self._trim_release)
        cr = tk.Frame(self._trim_tab, bg=BG_COLOR); cr.pack(fill=tk.X, padx=10, pady=(4, 8))
        tc = tk.Frame(cr, bg=BG_COLOR); tc.pack(side=tk.LEFT)
        self._trim_tc_start = tk.Label(tc, text="IN  00:00:00.00",
                                       bg=BG_COLOR, fg="#00FF88",
                                       font=("Consolas", 9, "bold"))
        self._trim_tc_start.pack(side=tk.LEFT)
        tk.Label(tc, text="   \u2192   ", bg=BG_COLOR, fg="#555",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._trim_tc_end = tk.Label(tc, text="OUT 00:00:00.00",
                                     bg=BG_COLOR, fg="#FF6040",
                                     font=("Consolas", 9, "bold"))
        self._trim_tc_end.pack(side=tk.LEFT)
        tk.Label(tc, text="   |   ", bg=BG_COLOR, fg="#555",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._trim_lbl_dur = tk.Label(tc, text="Duration: --  |  -- frames",
                                      bg=BG_COLOR, fg=TEXT_COLOR, font=("Segoe UI", 9))
        self._trim_lbl_dur.pack(side=tk.LEFT)
        jf = tk.Frame(cr, bg=BG_COLOR); jf.pack(side=tk.RIGHT)
        tk.Label(jf, text="Jump %:", bg=BG_COLOR, fg="#555",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 2))
        self._trim_jump = tk.StringVar(value="0")
        tk.Entry(jf, textvariable=self._trim_jump, width=6,
                 bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                 font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(jf, text="Set IN",  command=lambda: self._trim_jump_set("start"),
                  bg=BTN_COLOR, fg="#00FF88", relief=tk.FLAT,
                  font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(jf, text="Set OUT", command=lambda: self._trim_jump_set("end"),
                  bg=BTN_COLOR, fg="#FF6040", relief=tk.FLAT,
                  font=("Segoe UI", 8)).pack(side=tk.LEFT)

    # ---- trim helpers --------------------------------------------------------

    def _trim_auto_load(self):
        if self.state.video and self.state.video.path:
            self._trim_load_path(self.state.video.path)

    def _trim_load(self):
        src = (self.state.video.path
               if self.state.video and self.state.video.path else "")
        if not src:
            src = filedialog.askopenfilename(
                title="Select video to trim",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4"),
                           ("All", "*.*")])
        if not src:
            return
        self._trim_load_path(src)

    def _trim_load_path(self, src):
        try:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self._status("Trim: cannot open source.")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_  = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fw_   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh_   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read(); cap.release()
            if not ret:
                self._status("Trim: could not read first frame.")
                return
            self.state.trim_source_path = src
            self.state.trim_total_fr    = total
            self.state.trim_fps         = fps_
            self._trim_frame_bgr        = frame
            name = Path(src).name
            self._trim_lbl_src.config(
                text=f"{name}  {fw_}\u00d7{fh_}  {fps_:.1f} fps  {total:,} fr")
            self._trim_hint.place_forget()
            self._trim_btn_play.config(state=tk.NORMAL)
            self._trim_btn_save.config(state=tk.NORMAL)
            self._trim_blit()
            self._trim_update_readouts()
            self._trim_draw_slider()
            self._status(f"Trim: loaded {name}  ({total:,} frames  {fps_:.1f} fps)")
        except Exception as e:
            self._status(f"Trim load error: {e}")

    def _trim_reset(self):
        self._trim_stop_play()
        self._trim_start_pct.set(0.0); self._trim_end_pct.set(100.0)
        self._trim_play_pos.set(0.0)
        self._trim_draw_slider(); self._trim_update_readouts()
        if self.state.trim_total_fr > 0:
            self._trim_seek(0.0)

    def _trim_toggle_play(self):
        if self._trim_playing: self._trim_stop_play()
        else: self._trim_start_play()

    def _trim_start_play(self):
        if self.state.trim_total_fr == 0: return
        self._trim_playing = True
        self._trim_btn_play.config(text="\u25a0  Stop")
        threading.Thread(target=self._trim_play_thread_fn, daemon=True).start()

    def _trim_stop_play(self):
        self._trim_playing = False
        if hasattr(self, "_trim_btn_play"):
            self._trim_btn_play.config(text="\u25b6  Play")
        if self._trim_play_cap is not None:
            try: self._trim_play_cap.release()
            except Exception: pass
            self._trim_play_cap = None

    def _trim_play_thread_fn(self):
        src = self.state.trim_source_path
        if not src: self._trim_playing = False; return
        try:
            cap = cv2.VideoCapture(src)
            self._trim_play_cap = cap
            total = self.state.trim_total_fr
            fps_  = self.state.trim_fps or 30.0
            spf   = 1.0 / fps_
            start_fr = max(0, int(self._trim_play_pos.get() / 100.0 * total))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_fr); fr_idx = start_fr
            while self._trim_playing and fr_idx < total:
                t0 = _time.perf_counter()
                ret, frame = cap.read()
                if not ret: break
                fr_idx += 1; pct = fr_idx / total * 100.0
                self._trim_play_pos.set(pct)
                fc = frame.copy()
                self.root.after(0, lambda f=fc, p=pct: self._trim_play_update(f, p))
                _time.sleep(max(0.0, spf - (_time.perf_counter() - t0)))
        except Exception: pass
        finally:
            try: cap.release()
            except Exception: pass
            self._trim_play_cap = None; self._trim_playing = False
            self.root.after(0, lambda: self._trim_btn_play.config(text="\u25b6  Play"))

    def _trim_play_update(self, frame, pct):
        if frame is None: return
        self._trim_frame_bgr = frame
        self._trim_blit(); self._trim_draw_slider()
        dur = (self.state.trim_total_fr / self.state.trim_fps
               if self.state.trim_fps > 0 else 0.0)
        self._trim_lbl_pos.config(text=self._trim_tc(pct / 100.0 * dur))

    def _trim_save_and_load(self):
        self._trim_stop_play()
        src = self.state.trim_source_path
        if not src or self.state.trim_total_fr == 0:
            messagebox.showwarning("No source", "Load a video first."); return
        s_pct = self._trim_start_pct.get(); e_pct = self._trim_end_pct.get()
        fps_  = self.state.trim_fps or 30.0; total = self.state.trim_total_fr
        sf = int(s_pct / 100.0 * total); ef = int(e_pct / 100.0 * total)
        if ef <= sf:
            messagebox.showwarning("Invalid range", "OUT must be after IN."); return
        src_path  = Path(src)
        suggested = src_path.parent / (src_path.stem + "_trimmed" + src_path.suffix)
        out_str = filedialog.asksaveasfilename(
            title="Save trimmed video",
            initialdir=str(src_path.parent),
            initialfile=suggested.name,
            defaultextension=src_path.suffix,
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi"),
                       ("MOV", "*.mov"), ("All", "*.*")])
        if not out_str: return
        self._status(f"Trim: saving frames {sf}\u2013{ef} \u2192 {Path(out_str).name} ...")
        self._trim_btn_save.config(state=tk.DISABLED)
        self._trim_btn_play.config(state=tk.DISABLED)
        def _export():
            try:
                cap    = cv2.VideoCapture(src)
                w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(
                    out_str, cv2.VideoWriter_fourcc(*"mp4v"), fps_, (w, h))
                cap.set(cv2.CAP_PROP_POS_FRAMES, sf); written = 0
                for _ in range(ef - sf):
                    ret, frame = cap.read()
                    if not ret: break
                    writer.write(frame); written += 1
                cap.release(); writer.release()
                self.root.after(0, lambda: self._trim_export_done(out_str, written))
            except Exception as e:
                self.root.after(0, lambda: self._status(f"Trim save error: {e}"))
                self.root.after(0, lambda: self._trim_btn_save.config(state=tk.NORMAL))
                self.root.after(0, lambda: self._trim_btn_play.config(state=tk.NORMAL))
        threading.Thread(target=_export, daemon=True).start()

    def _trim_export_done(self, out_path, frames_written):
        self._status(f"Trim saved: {Path(out_path).name}  ({frames_written:,} frames)")
        self._trim_btn_save.config(state=tk.NORMAL)
        self._trim_btn_play.config(state=tk.NORMAL)
        self._trim_start_pct.set(0.0); self._trim_end_pct.set(100.0)
        self.state.trim_source_path = out_path
        self._trim_load_path(out_path)
        if messagebox.askyesno(
                "Trim saved",
                f"Saved {frames_written:,} frames to:\n{Path(out_path).name}\n\n"
                "Load trimmed video into the tracker now?"):
            self._load_video(out_path)
            self.notebook.select(self._setup_tab)

    def _trim_blit(self):
        if self._trim_frame_bgr is None: return
        c = self._trim_canvas
        cw = c.winfo_width(); ch = c.winfo_height()
        if cw < 4 or ch < 4: return
        fh_, fw_ = self._trim_frame_bgr.shape[:2]
        scale = min(cw / fw_, ch / fh_)
        nw = int(fw_ * scale); nh = int(fh_ * scale)
        ox = (cw - nw) // 2; oy = (ch - nh) // 2
        resized = cv2.resize(self._trim_frame_bgr, (nw, nh),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        c.delete("all"); c.create_image(ox, oy, anchor=tk.NW, image=img)
        self._trim_photo = img
        s = self._trim_start_pct.get(); e = self._trim_end_pct.get()
        c.create_rectangle(0, ch - 20, cw, ch, fill="#000", outline="")
        c.create_text(8, ch - 10, anchor="w",
                      text=f"IN {s:.2f}%   OUT {e:.2f}%",
                      fill=ACCENT_COLOR, font=("Consolas", 8, "bold"))
        in_x  = ox + int(nw * s / 100.0)
        out_x = ox + int(nw * e / 100.0)
        c.create_line(in_x,  oy, in_x,  oy + nh, fill="#00FF88", width=2, dash=(5, 3))
        c.create_line(out_x, oy, out_x, oy + nh, fill="#FF6040", width=2, dash=(5, 3))
        c.create_text(in_x  + 4, oy + 8, anchor="w",
                      text="IN",  fill="#00FF88", font=("Consolas", 8, "bold"))
        c.create_text(out_x + 4, oy + 8, anchor="w",
                      text="OUT", fill="#FF6040", font=("Consolas", 8, "bold"))

    def _trim_seek(self, pct):
        src = self.state.trim_source_path
        if not src or self.state.trim_total_fr == 0: return
        try:
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    max(0, int(pct / 100.0 * self.state.trim_total_fr) - 1))
            ret, frame = cap.read(); cap.release()
            if ret: self._trim_frame_bgr = frame; self._trim_blit()
        except Exception: pass

    def _trim_draw_slider(self):
        c = self._trim_sc; c.delete("all")
        cw = c.winfo_width(); ch = c.winfo_height()
        if cw < 20: return
        HR = 13; PAD = HR + 6; TY = ch // 2 - 6; usable = cw - 2 * PAD
        def _px(p): return int(PAD + (p / 100.0) * usable)
        s_pct = self._trim_start_pct.get(); e_pct = self._trim_end_pct.get()
        pl_pct = self._trim_play_pos.get()
        sx = _px(s_pct); ex = _px(e_pct); plx = _px(pl_pct)
        c.create_rectangle(PAD, TY - 5, cw - PAD, TY + 5, fill="#111120", outline="#333")
        c.create_rectangle(sx, TY - 5, ex, TY + 5, fill=ACCENT_COLOR, outline="")
        for t in range(0, 101, 5):
            tx = _px(t); major = (t % 25 == 0)
            c.create_line(tx, TY + 7, tx, TY + (14 if major else 9),
                          fill="#555" if not major else "#888", width=1)
            if major:
                c.create_text(tx, TY + 22, text=f"{t}%", fill="#666",
                              font=("Segoe UI", 7))
        if self._trim_playing or pl_pct > 0:
            c.create_line(plx, TY - 10, plx, TY + 10, fill="#ffffff", width=2)
            c.create_polygon(plx - 5, TY - 10, plx + 5, TY - 10, plx, TY - 5,
                             fill="#ffffff", outline="")
        c.create_oval(sx - HR, TY - HR, sx + HR, TY + HR,
                      fill="#00FF88", outline="#ffffff", width=2)
        c.create_text(sx, TY, text="I", fill="#000", font=("Segoe UI", 8, "bold"))
        c.create_text(sx, TY - HR - 6, text=f"{s_pct:.1f}%",
                      fill="#00FF88", font=("Segoe UI", 7))
        c.create_oval(ex - HR, TY - HR, ex + HR, TY + HR,
                      fill="#FF6040", outline="#ffffff", width=2)
        c.create_text(ex, TY, text="O", fill="#fff", font=("Segoe UI", 8, "bold"))
        c.create_text(ex, TY - HR - 6, text=f"{e_pct:.1f}%",
                      fill="#FF6040", font=("Segoe UI", 7))

    def _trim_press(self, event):
        c = self._trim_sc; cw = c.winfo_width(); HR = 13; PAD = HR + 6; usable = cw - 2 * PAD
        def _px(p): return int(PAD + (p / 100.0) * usable)
        ds = abs(event.x - _px(self._trim_start_pct.get()))
        de = abs(event.x - _px(self._trim_end_pct.get()))
        self._trim_drag = "start" if ds <= de else "end"

    def _trim_drag_move(self, event):
        if not self._trim_drag: return
        c = self._trim_sc; cw = c.winfo_width(); HR = 13; PAD = HR + 6; usable = cw - 2 * PAD
        pct = max(0.0, min(100.0, (event.x - PAD) / usable * 100.0))
        if self._trim_drag == "start":
            pct = min(pct, self._trim_end_pct.get() - 0.5)
            self._trim_start_pct.set(round(pct, 2))
        else:
            pct = max(pct, self._trim_start_pct.get() + 0.5)
            self._trim_end_pct.set(round(pct, 2))
        self._trim_draw_slider(); self._trim_update_readouts(); self._trim_blit()

    def _trim_release(self, event):
        if self._trim_drag:
            pct = (self._trim_start_pct.get()
                   if self._trim_drag == "start"
                   else self._trim_end_pct.get())
            self._trim_seek(pct)
        self._trim_drag = None

    def _trim_jump_set(self, which):
        try: pct = max(0.0, min(100.0, float(self._trim_jump.get())))
        except ValueError: return
        if which == "start":
            pct = min(pct, self._trim_end_pct.get() - 0.5)
            self._trim_start_pct.set(round(pct, 2))
        else:
            pct = max(pct, self._trim_start_pct.get() + 0.5)
            self._trim_end_pct.set(round(pct, 2))
        self._trim_draw_slider(); self._trim_update_readouts(); self._trim_seek(pct)

    def _trim_tc(self, sec):
        h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
        return f"{h:02d}:{m:02d}:{s:05.2f}"

    def _trim_update_readouts(self):
        s = self._trim_start_pct.get(); e = self._trim_end_pct.get()
        fps_ = self.state.trim_fps; tf = self.state.trim_total_fr
        dur = tf / fps_ if fps_ > 0 else 0.0
        s_sec = (s / 100.0) * dur; e_sec = (e / 100.0) * dur
        sel_f = int((e - s) / 100.0 * tf)
        self._trim_tc_start.config(text=f"IN  {self._trim_tc(s_sec)}")
        self._trim_tc_end.config(  text=f"OUT {self._trim_tc(e_sec)}")
        self._trim_lbl_dur.config( text=f"Duration: {self._trim_tc(e_sec - s_sec)}  |  {sel_f:,} frames")

    # =========================================================================
    # Calibrate Tab
    # =========================================================================

    def _build_calib_tab(self):
        # Face-ID name bar
        _id_bar3 = tk.Frame(self._calib_tab, bg="#0d1117")
        _id_bar3.pack(fill=tk.X, side=tk.TOP)
        make_name_label(_id_bar3).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(_id_bar3, text="[ID] Identify Athlete",
                  command=self._open_identify_popup,
                  bg="#21262d", fg="#e0e0e0", relief=tk.FLAT,
                  font=("Segoe UI", 8), padx=10, pady=3).pack(side=tk.RIGHT, padx=4)
        # Canvas (left)
        self._calib_canvas = tk.Canvas(self._calib_tab, bg="#1a1a1a",
                                       cursor="crosshair",
                                       highlightthickness=1,
                                       highlightbackground=ACCENT_COLOR)
        self._calib_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                padx=(6, 2), pady=6)
        self._calib_canvas.bind("<Button-1>",  self._calib_left_click)
        self._calib_canvas.bind("<Button-3>",  self._calib_right_click)
        self._calib_canvas.bind("<Configure>", lambda e: self._refresh_calib())
        self._calib_canvas.bind("<Motion>",    lambda e: self._on_mag_motion(e, "calib"))
        self._calib_canvas.bind("<Leave>",     lambda e: self._on_mag_leave("calib"))
        self._bind_canvas_nav(self._calib_canvas)

        # Right sidebar — scrollable
        _r_outer = tk.Frame(self._calib_tab, bg=PANEL_BG, width=320)
        _r_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 6), pady=6)
        _r_outer.pack_propagate(False)
        _r_sb = tk.Scrollbar(_r_outer, orient=tk.VERTICAL,
                             bg=PANEL_BG, troughcolor="#0d1117")
        _r_sb.pack(side=tk.RIGHT, fill=tk.Y)
        _r_cv = tk.Canvas(_r_outer, bg=PANEL_BG, highlightthickness=0,
                          yscrollcommand=_r_sb.set)
        _r_cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _r_sb.config(command=_r_cv.yview)
        right = tk.Frame(_r_cv, bg=PANEL_BG)
        _r_win_id = _r_cv.create_window((0, 0), window=right, anchor=tk.NW)
        def _r_resize(ev, cv=_r_cv, wid=_r_win_id):
            cv.itemconfig(wid, width=ev.width)
        def _r_content(ev, cv=_r_cv):
            cv.configure(scrollregion=cv.bbox("all"))
        def _r_wheel(ev, cv=_r_cv):
            cv.yview_scroll(int(-1*(ev.delta/120)), "units")
        _r_cv.bind("<Configure>", _r_resize)
        right.bind("<Configure>", _r_content)
        _r_cv.bind("<MouseWheel>", _r_wheel)
        right.bind("<MouseWheel>", _r_wheel)

        # Mode selector
        mode_f = tk.Frame(right, bg=PANEL_BG)
        mode_f.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(mode_f, text="Mode:", fg=ACCENT_COLOR, bg=PANEL_BG,
                 font=("Consolas", 9, "bold")).pack(side=tk.LEFT)
        self._calib_mag_btn = tk.Button(
            mode_f, text="Magnify", command=self._toggle_magnifier,
            bg=BTN_COLOR, fg=TEXT_COLOR, relief=tk.FLAT,
            padx=6, pady=2, font=("Segoe UI", 8), cursor="hand2")
        self._calib_mag_btn.pack(side=tk.RIGHT, padx=2)
        self._calib_mode_var = tk.StringVar(value="linear")
        tk.Radiobutton(mode_f, text="Linear (2 pts)",
                       variable=self._calib_mode_var, value="linear",
                       bg=PANEL_BG, fg=TEXT_COLOR, selectcolor=PANEL_BG,
                       activebackground=PANEL_BG, font=("Segoe UI", 9),
                       command=self._on_calib_mode_change).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(mode_f, text="Field (4 pts)",
                       variable=self._calib_mode_var, value="field",
                       bg=PANEL_BG, fg=TEXT_COLOR, selectcolor=PANEL_BG,
                       activebackground=PANEL_BG, font=("Segoe UI", 9),
                       command=self._on_calib_mode_change).pack(side=tk.LEFT, padx=4)

        # Click counter label
        self._calib_click_lbl = tk.Label(right,
            text="Click 2 points on the frame to define calibration line.",
            fg="#888", bg=PANEL_BG, font=("Segoe UI", 8), wraplength=290,
            justify=tk.LEFT)
        self._calib_click_lbl.pack(padx=8, pady=(0, 4), anchor=tk.W)

        # LINEAR section
        tk.Label(right, text="LINEAR CALIBRATION", fg=ACCENT_COLOR, bg=PANEL_BG,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(6, 2))
        lin_inp = tk.Frame(right, bg=PANEL_BG)
        lin_inp.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(lin_inp, text="Name:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._lin_name_var = tk.StringVar(value="Calib1")
        tk.Entry(lin_inp, textvariable=self._lin_name_var, width=10,
                 bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                 font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Label(lin_inp, text="Distance (m):", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 0))
        self._lin_dist_var = tk.StringVar(value="1.00")
        tk.Entry(lin_inp, textvariable=self._lin_dist_var, width=6,
                 bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                 font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        _make_btn(right, "+ Add Linear Calibration", self._add_linear_calib,
                  bg=ACCENT_COLOR, fg="#000", side=tk.TOP, padx=8)
        lb_lin = tk.Frame(right, bg=PANEL_BG)
        lb_lin.pack(fill=tk.X, padx=8, pady=2)
        sb_lin = tk.Scrollbar(lb_lin)
        sb_lin.pack(side=tk.RIGHT, fill=tk.Y)
        self._lin_listbox = tk.Listbox(lb_lin, bg="#0d1117", fg=TEXT_COLOR,
                                       selectbackground="#1f6feb",
                                       font=("Consolas", 8), height=4,
                                       yscrollcommand=sb_lin.set, relief=tk.FLAT)
        self._lin_listbox.pack(fill=tk.X, expand=True)
        sb_lin.config(command=self._lin_listbox.yview)
        _make_btn(right, "Remove Linear", self._remove_linear_calib,
                  side=tk.TOP, padx=8)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # FIELD section
        tk.Label(right, text="FIELD CALIBRATION", fg=ACCENT_COLOR, bg=PANEL_BG,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(2, 2))
        fld_inp = tk.Frame(right, bg=PANEL_BG)
        fld_inp.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(fld_inp, text="Name:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._fld_name_var = tk.StringVar(value="Field1")
        tk.Entry(fld_inp, textvariable=self._fld_name_var, width=9,
                 bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                 font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Label(fld_inp, text="W(m):", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._fld_w_var = tk.StringVar(value="5.00")
        tk.Entry(fld_inp, textvariable=self._fld_w_var, width=5,
                 bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                 font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Label(fld_inp, text="H(m):", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._fld_h_var = tk.StringVar(value="5.00")
        tk.Entry(fld_inp, textvariable=self._fld_h_var, width=5,
                 bg="#0d1117", fg=TEXT_COLOR, insertbackground=TEXT_COLOR,
                 font=("Segoe UI", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        tk.Label(right,
                 text="Click corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left",
                 fg="#888", bg=PANEL_BG, font=("Segoe UI", 7),
                 wraplength=290).pack(padx=8, anchor=tk.W)
        _make_btn(right, "+ Add Field Calibration", self._add_field_calib,
                  bg=ACCENT_COLOR, fg="#000", side=tk.TOP, padx=8)
        lb_fld = tk.Frame(right, bg=PANEL_BG)
        lb_fld.pack(fill=tk.X, padx=8, pady=2)
        sb_fld = tk.Scrollbar(lb_fld)
        sb_fld.pack(side=tk.RIGHT, fill=tk.Y)
        self._fld_listbox = tk.Listbox(lb_fld, bg="#0d1117", fg=TEXT_COLOR,
                                       selectbackground="#1f6feb",
                                       font=("Consolas", 8), height=4,
                                       yscrollcommand=sb_fld.set, relief=tk.FLAT)
        self._fld_listbox.pack(fill=tk.X, expand=True)
        sb_fld.config(command=self._fld_listbox.yview)
        _make_btn(right, "Remove Field", self._remove_field_calib,
                  side=tk.TOP, padx=8)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ASSIGNMENT section
        tk.Label(right, text="ASSIGN TO POINTS", fg=ACCENT_COLOR, bg=PANEL_BG,
                 font=("Consolas", 9, "bold")).pack(anchor=tk.W, padx=8, pady=(2, 2))
        assign_top = tk.Frame(right, bg=PANEL_BG)
        assign_top.pack(fill=tk.X, padx=8, pady=2)
        tk.Label(assign_top, text="Point:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self._assign_pt_var = tk.StringVar()
        self._assign_pt_cb  = ttk.Combobox(assign_top,
                                            textvariable=self._assign_pt_var,
                                            width=7, state="readonly",
                                            font=("Segoe UI", 8))
        self._assign_pt_cb.pack(side=tk.LEFT, padx=4)
        tk.Label(assign_top, text="Calib:", fg=TEXT_COLOR, bg=PANEL_BG,
                 font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(4, 0))
        self._assign_cal_var = tk.StringVar()
        self._assign_cal_cb  = ttk.Combobox(assign_top,
                                             textvariable=self._assign_cal_var,
                                             width=12, state="readonly",
                                             font=("Segoe UI", 8))
        self._assign_cal_cb.pack(side=tk.LEFT, padx=4)
        _make_btn(right, "Assign", self._do_assign_calib,
                  bg=ACCENT_COLOR, fg="#000", side=tk.TOP, padx=8)
        lb_asgn = tk.Frame(right, bg=PANEL_BG)
        lb_asgn.pack(fill=tk.BOTH, expand=True, padx=8, pady=2)
        sb_asgn = tk.Scrollbar(lb_asgn)
        sb_asgn.pack(side=tk.RIGHT, fill=tk.Y)
        self._assign_listbox = tk.Listbox(lb_asgn, bg="#0d1117", fg=TEXT_COLOR,
                                          selectbackground="#1f6feb",
                                          font=("Consolas", 8), height=4,
                                          yscrollcommand=sb_asgn.set, relief=tk.FLAT)
        self._assign_listbox.pack(fill=tk.BOTH, expand=True)
        sb_asgn.config(command=self._assign_listbox.yview)
        _make_btn(right, "Clear Assignment", self._clear_assignment,
                  side=tk.TOP, padx=8)

    def _on_calib_mode_change(self):
        self._calib_mode = self._calib_mode_var.get()
        self._calib_click_pts = []
        self._update_calib_click_label()
        self._refresh_calib()

    def _update_calib_click_label(self):
        mode = self._calib_mode
        n    = len(self._calib_click_pts)
        need = 2 if mode == "linear" else 4
        hint = ("Click 2 points on the frame to define calibration line."
                if mode == "linear" else
                "Click 4 corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
        if n == 0:
            self._calib_click_lbl.config(text=hint)
        else:
            self._calib_click_lbl.config(
                text=f"{n}/{need} points placed. {need-n} more needed.")

    def _calib_left_click(self, event):
        if not self.state.frames:
            return
        xn, yn = self._canvas_to_norm(event.x, event.y, self.state, "calib")
        limit  = 2 if self._calib_mode == "linear" else 4
        if len(self._calib_click_pts) >= limit:
            self._calib_click_pts = []
        self._calib_click_pts.append((xn, yn))
        self._update_calib_click_label()
        self._refresh_calib()

    def _calib_right_click(self, _event):
        if self._calib_click_pts:
            self._calib_click_pts.pop()
            self._update_calib_click_label()
            self._refresh_calib()

    def _refresh_calib(self):
        if not self.state.frames:
            return
        idx   = self.state.current_frame
        idx   = max(0, min(idx, len(self.state.frames) - 1))
        frame = self.state.frames[idx].copy()
        H, W  = frame.shape[:2]
        mode  = self._calib_mode

        # Draw saved calibrations
        for cal in self.state.calib_mgr.linear:
            x0 = int(cal.p0_norm[0]*W); y0 = int(cal.p0_norm[1]*H)
            x1 = int(cal.p1_norm[0]*W); y1 = int(cal.p1_norm[1]*H)
            cv2.line(frame, (x0,y0), (x1,y1), (0,200,150), 2, cv2.LINE_AA)
            cv2.circle(frame, (x0,y0), 5, (0,200,150), -1, cv2.LINE_AA)
            cv2.circle(frame, (x1,y1), 5, (0,200,150), -1, cv2.LINE_AA)
            cv2.putText(frame, f"L:{cal.name} {cal.real_m:.2f}m",
                        (x0, y0-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0,200,150), 1, cv2.LINE_AA)
        for cal in self.state.calib_mgr.field:
            pts = [(int(c[0]*W), int(c[1]*H)) for c in cal.corners_norm]
            if len(pts) == 4:
                for j in range(4):
                    cv2.line(frame, pts[j], pts[(j+1)%4], (255,200,0), 2, cv2.LINE_AA)
                    cv2.circle(frame, pts[j], 5, (255,200,0), -1, cv2.LINE_AA)
                lbl_pts = ["TL","TR","BR","BL"]
                for j, pt in enumerate(pts):
                    cv2.putText(frame, lbl_pts[j], (pt[0]+5, pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,200,0), 1, cv2.LINE_AA)
                cx = sum(p[0] for p in pts)//4
                cy = sum(p[1] for p in pts)//4
                cv2.putText(frame, f"F:{cal.name} {cal.real_w}x{cal.real_h}m",
                            (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255,200,0), 1, cv2.LINE_AA)

        # Draw pending click points
        colors_p = [(0,150,255),(0,220,150),(255,165,0),(200,0,255)]
        for i, (xn, yn) in enumerate(self._calib_click_pts):
            cx, cy = int(xn*W), int(yn*H)
            col = colors_p[i % len(colors_p)]
            cv2.circle(frame, (cx,cy), 6, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx,cy), 9, col,  1, cv2.LINE_AA)
            lbl = ["TL","TR","BR","BL"][i] if mode == "field" else str(i+1)
            cv2.putText(frame, lbl, (cx+8, cy-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            if mode == "linear" and i == 1 and len(self._calib_click_pts) >= 2:
                p0x = int(self._calib_click_pts[0][0]*W)
                p0y = int(self._calib_click_pts[0][1]*H)
                cv2.line(frame, (p0x,p0y), (cx,cy), (0,150,255), 1, cv2.LINE_AA)

        self._blit_to_canvas(self._calib_canvas, frame, self.state, "calib")

    def _add_linear_calib(self):
        if len(self._calib_click_pts) < 2:
            messagebox.showwarning("Need 2 Points",
                "Click exactly 2 points on the frame first (Linear mode).")
            return
        try:
            dist = float(self._lin_dist_var.get())
            assert dist > 0
        except Exception:
            messagebox.showwarning("Invalid Distance",
                "Enter a positive number for distance (metres).")
            return
        name = self._lin_name_var.get().strip() or f"Lin{len(self.state.calib_mgr.linear)+1}"
        cal  = LinearCalibration(name=name,
                                  p0_norm=self._calib_click_pts[0],
                                  p1_norm=self._calib_click_pts[1],
                                  real_m=round(dist, 4))
        self.state.calib_mgr.add_linear(cal)
        fw = self.state.video.width  if self.state.video else 960
        fh = self.state.video.height if self.state.video else 540
        ppm = cal.px_per_m(fw, fh)
        self._lin_listbox.insert(tk.END,
            f"  {name}  {dist:.2f}m  ({ppm:.1f}px/m)")
        self._lin_name_var.set(f"Calib{len(self.state.calib_mgr.linear)+1}")
        self._calib_click_pts = []
        self._update_calib_click_label()
        self._update_assign_combos()
        self._refresh_calib()
        self._status(f"Linear calibration '{name}' added  ({ppm:.1f} px/m).")

    def _add_field_calib(self):
        if len(self._calib_click_pts) < 4:
            messagebox.showwarning("Need 4 Points",
                "Click exactly 4 corner points (Field mode).")
            return
        try:
            w = float(self._fld_w_var.get())
            h = float(self._fld_h_var.get())
            assert w > 0 and h > 0
        except Exception:
            messagebox.showwarning("Invalid Dimensions",
                "Enter positive W and H values in metres.")
            return
        name = self._fld_name_var.get().strip() or f"Field{len(self.state.calib_mgr.field)+1}"
        cal  = FieldCalibration(name=name,
                                 corners_norm=list(self._calib_click_pts[:4]),
                                 real_w=round(w,4), real_h=round(h,4))
        self.state.calib_mgr.add_field(cal)
        self._fld_listbox.insert(tk.END, f"  {name}  {w:.2f}x{h:.2f}m")
        self._fld_name_var.set(f"Field{len(self.state.calib_mgr.field)+1}")
        self._calib_click_pts = []
        self._update_calib_click_label()
        self._update_assign_combos()
        self._refresh_calib()
        self._status(f"Field calibration '{name}' added  ({w:.2f}x{h:.2f}m).")

    def _remove_linear_calib(self):
        sel = self._lin_listbox.curselection()
        if not sel: return
        i = sel[0]
        self.state.calib_mgr.remove_linear(i)
        self._lin_listbox.delete(i)
        self._update_assign_combos()
        self._refresh_calib()

    def _remove_field_calib(self):
        sel = self._fld_listbox.curselection()
        if not sel: return
        i = sel[0]
        self.state.calib_mgr.remove_field(i)
        self._fld_listbox.delete(i)
        self._update_assign_combos()
        self._refresh_calib()

    def _update_assign_combos(self):
        self._assign_pt_cb["values"]  = self.state.point_labels
        self._assign_cal_cb["values"] = self.state.calib_mgr.all_names()

    def _do_assign_calib(self):
        pt_lbl  = self._assign_pt_var.get()
        cal_lbl = self._assign_cal_var.get()
        if not pt_lbl or not cal_lbl:
            messagebox.showwarning("Select Both",
                "Choose a point and a calibration to assign.")
            return
        if pt_lbl not in self.state.point_labels:
            return
        pt_idx    = self.state.point_labels.index(pt_lbl)
        all_names = self.state.calib_mgr.all_names()
        if cal_lbl not in all_names:
            return
        flat_idx = all_names.index(cal_lbl)
        n_lin    = len(self.state.calib_mgr.linear)
        if flat_idx < n_lin:
            self.state.calib_mgr.assign(pt_idx, "linear", flat_idx)
            tag = "L"
        else:
            self.state.calib_mgr.assign(pt_idx, "field", flat_idx - n_lin)
            tag = "F"
        disp = f"  {pt_lbl}  ->  [{tag}] {cal_lbl[3:]}"
        for i in range(self._assign_listbox.size()):
            if self._assign_listbox.get(i).strip().startswith(pt_lbl):
                self._assign_listbox.delete(i)
                break
        self._assign_listbox.insert(tk.END, disp)
        n = len(self.state.calib_mgr.point_assignment)
        if hasattr(self, "_kin_status_lbl"):
            self._kin_status_lbl.config(
                text=f"{n} point(s) assigned. Click Compute & Export.")
        self._status(f"'{pt_lbl}' assigned to [{tag}] {cal_lbl[3:]}.")

    def _clear_assignment(self):
        sel = self._assign_listbox.curselection()
        if not sel: return
        i    = sel[0]
        text = self._assign_listbox.get(i).strip()
        pt_lbl = text.split("->")[0].strip()
        if pt_lbl in self.state.point_labels:
            self.state.calib_mgr.unassign(
                self.state.point_labels.index(pt_lbl))
        self._assign_listbox.delete(i)

    # =========================================================================
    # Kinematics
    # =========================================================================

    def _compute_export_kinematics(self):
        if not self.state.tracking_done or self.state.result is None:
            messagebox.showwarning("No Tracking", "Run tracking first.")
            return
        if not self.state.calib_mgr.has_any():
            messagebox.showwarning("No Calibrations",
                "Add calibrations in the Calibrate tab first.")
            return
        if not self.state.calib_mgr.point_assignment:
            messagebox.showwarning("No Assignments",
                "Assign points to calibrations in the Calibrate tab first.")
            return
        tracks = (self.state.smoothed_tracks
                  if self.state.smoothed_tracks is not None
                  else self.state.result.tracks)
        fw  = self.state.video.width
        fh  = self.state.video.height
        fps = self.state.video.fps
        win  = self._kin_win_var.get()  if hasattr(self, "_kin_win_var")  else 11
        poly = self._kin_poly_var.get() if hasattr(self, "_kin_poly_var") else 3
        self._status("Computing kinematics...")
        try:
            kin = compute_kinematics(
                tracks_px  = tracks,
                occlusions = self.state.result.occlusions,
                calib_mgr  = self.state.calib_mgr,
                frame_w    = fw, frame_h = fh,
                fps        = fps,
                sg_window  = win, sg_poly = poly,
            )
            out = export_kinematics_csv(
                kinematics = kin, fps = fps,
                video_name = self.state.video.path,
                labels     = self.state.point_labels,
            )
            n = len(self.state.calib_mgr.point_assignment)
            self._status(f"Kinematics exported -> {out}")
            if hasattr(self, "_kin_status_lbl"):
                self._kin_status_lbl.config(
                    text=f"Done — {n} pt(s). {os.path.basename(out)}")
            messagebox.showinfo("Kinematics Exported",
                f"Kinematics computed for {n} point(s).\n\n"
                f"Columns: world_x/y_m, vel_x/y_ms,\n"
                f"         speed_ms, accel_ms2, distance_m\n\n"
                f"File:\n  {out}")
        except Exception as ex:
            err = str(ex)
            self._status(f"Kinematics error: {err}")
            logger.exception("Kinematics failed")
            messagebox.showerror("Kinematics Error", err)

    def _open_identify_popup(self):
        """Grab current frame and open the Identify popup."""
        try:
            from identify_popup_tapir import show_identify_popup
            frame = grab_frame_tapir(self)
            show_identify_popup(self.root, self.face_mgr, frame, app_root=APP_ROOT)
        except Exception as exc:
            self._status(f"Face ID popup error: {exc}")

    def _on_tab_changed(self, _event=None):
        try:
            tab  = self.notebook.select()
            name = self.notebook.tab(tab, 'text').strip()
            if name == 'Calibrate':
                self.root.after(50, self._refresh_calib)
            elif name == 'Trim':
                if (self.state.video and self.state.video.path
                        and self.state.video.path != self.state.trim_source_path):
                    self.root.after(100, self._trim_auto_load)
        except Exception:
            pass


    # =========================================================================
    # Magnifier / Loupe
    # =========================================================================

    def _toggle_magnifier(self):
        """Toggle magnifier on/off; sync button appearance on both tabs."""
        self._magnifier_on = not self._magnifier_on
        label = "Magnify ON" if self._magnifier_on else "Magnify"
        bg    = ACCENT_COLOR  if self._magnifier_on else BTN_COLOR
        fg    = "#000"        if self._magnifier_on else TEXT_COLOR
        for btn_attr in ("_setup_mag_btn", "_calib_mag_btn"):
            btn = getattr(self, btn_attr, None)
            if btn:
                btn.config(text=label, bg=bg, fg=fg)
        if not self._magnifier_on:
            for c in (getattr(self, "_setup_canvas", None),
                      getattr(self, "_calib_canvas", None)):
                if c:
                    c.delete("magnifier")

    def _on_mag_motion(self, event, mode: str):
        """Called on every mouse-move; renders loupe when magnifier is active."""
        if not self._magnifier_on or not self.state.frames:
            return
        canvas = self._setup_canvas if mode == "setup" else self._calib_canvas
        self._draw_magnifier(canvas, event.x, event.y, mode)

    def _on_mag_leave(self, mode: str):
        """Clear loupe when mouse leaves the canvas."""
        canvas = self._setup_canvas if mode == "setup" else self._calib_canvas
        canvas.delete("magnifier")

    def _draw_magnifier(self, canvas: tk.Canvas,
                        cx: int, cy: int, mode: str):
        if not self.state.frames:
            return

        idx   = max(0, min(self.state.current_frame, len(self.state.frames) - 1))
        frame = self.state.frames[idx].copy()
        fh, fw = frame.shape[:2]

        if mode == "setup":
            for i, (xn, yn) in enumerate(self.state.query_points):
                px, py = int(xn * fw), int(yn * fh)
                col = (self.state.point_colors[i]
                       if i < len(self.state.point_colors)
                       else COLORS[i % len(COLORS)])
                cv2.circle(frame, (px, py), config.POINT_RADIUS + 3, col, -1, cv2.LINE_AA)
        else:
            for cal in self.state.calib_mgr.linear:
                for xn, yn in [cal.p0_norm, cal.p1_norm]:
                    cv2.circle(frame, (int(xn*fw), int(yn*fh)), 4, (0,200,150), -1, cv2.LINE_AA)
            for cal in self.state.calib_mgr.field:
                for xn, yn in cal.corners_norm:
                    cv2.circle(frame, (int(xn*fw), int(yn*fh)), 4, (255,200,0), -1, cv2.LINE_AA)
            for xn, yn in self._calib_click_pts:
                cv2.circle(frame, (int(xn*fw), int(yn*fh)), 5, (0,150,255), -1, cv2.LINE_AA)

        if mode == "setup":
            scale = self.state.setup_scale
            ox, oy = self.state.setup_ox, self.state.setup_oy
        else:
            scale = self.state.calib_scale
            ox, oy = self.state.calib_ox, self.state.calib_oy

        if scale < 1e-6:
            return

        mx_frame = (cx - ox) / scale
        my_frame = (cy - oy) / scale

        LUPE_PX   = 180
        ZOOM_HALF = 25

        half_src = int(ZOOM_HALF / scale) + 1
        x0f = int(round(mx_frame)) - half_src
        y0f = int(round(my_frame)) - half_src
        x1f = x0f + half_src * 2
        y1f = y0f + half_src * 2

        pad_t = max(0, -y0f);   pad_b = max(0, y1f - fh)
        pad_l = max(0, -x0f);   pad_r = max(0, x1f - fw)
        x0c = max(0, x0f);  y0c = max(0, y0f)
        x1c = min(fw, x1f); y1c = min(fh, y1f)

        if x1c <= x0c or y1c <= y0c:
            return

        crop = frame[y0c:y1c, x0c:x1c]

        if any([pad_t, pad_b, pad_l, pad_r]):
            crop = cv2.copyMakeBorder(
                crop, pad_t, pad_b, pad_l, pad_r,
                cv2.BORDER_CONSTANT, value=(20, 20, 20))

        if crop.shape[0] < 2 or crop.shape[1] < 2:
            return

        loupe = cv2.resize(crop, (LUPE_PX, LUPE_PX), interpolation=cv2.INTER_LINEAR)

        mid  = LUPE_PX // 2
        GAP  = 8
        COL  = (0, 220, 150)
        COL2 = (255, 255, 255)

        cv2.line(loupe, (mid, 0),          (mid, mid - GAP),    COL, 1, cv2.LINE_AA)
        cv2.line(loupe, (mid, mid + GAP),  (mid, LUPE_PX - 1),  COL, 1, cv2.LINE_AA)
        cv2.line(loupe, (0, mid),          (mid - GAP, mid),    COL, 1, cv2.LINE_AA)
        cv2.line(loupe, (mid + GAP, mid),  (LUPE_PX - 1, mid),  COL, 1, cv2.LINE_AA)
        cv2.circle(loupe, (mid, mid), GAP - 1, COL,  1, cv2.LINE_AA)
        cv2.circle(loupe, (mid, mid), 1,        COL2, -1, cv2.LINE_AA)

        label = f"x={mx_frame:.1f}  y={my_frame:.1f}"
        cv2.rectangle(loupe, (0, LUPE_PX - 18), (LUPE_PX, LUPE_PX), (0, 0, 0), -1)
        cv2.putText(loupe, label, (5, LUPE_PX - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL, 1, cv2.LINE_AA)

        cv2.rectangle(loupe, (0, 0), (LUPE_PX - 1, LUPE_PX - 1), COL, 2, cv2.LINE_AA)

        rgb    = cv2.cvtColor(loupe, cv2.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        OFFSET = 20
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()

        lx = cx + OFFSET
        ly = cy + OFFSET

        if lx + LUPE_PX > cw - 4:
            lx = cx - OFFSET - LUPE_PX
        if ly + LUPE_PX > ch - 4:
            ly = cy - OFFSET - LUPE_PX

        lx = max(4, min(lx, cw - LUPE_PX - 4))
        ly = max(4, min(ly, ch - LUPE_PX - 4))

        canvas.delete("magnifier")
        canvas.create_image(lx, ly, anchor=tk.NW, image=tk_img, tags="magnifier")

        if mode == "setup":
            self._setup_mag_img = tk_img
        else:
            self._calib_mag_img = tk_img


    # =========================================================================
    # Engine selector
    # =========================================================================

    def _on_engine_change(self, _event=None):
        """Update status bar when engine dropdown changes."""
        status_map = {
            "CoTracker": "Engine: CoTracker  (joint point tracking)",
            "TAPIR":     "Engine: TAPIR  (per-point long-range tracking)",
            "Auto":      "Engine: Auto  (uses TAPIR if available, else CoTracker)",
        }
        self._status(status_map.get(self._engine_var.get(), "Engine changed."))

    def _text_dialog(self, title: str, default: str, prompt: str) -> str:
        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.configure(bg=BG_COLOR)
        dlg.resizable(False, False)
        var = tk.StringVar(value=default)
        tk.Label(dlg, text=prompt, fg=TEXT_COLOR,
                 bg=BG_COLOR).pack(padx=14, pady=(10, 2))
        entry = tk.Entry(dlg, textvariable=var, width=22,
                         bg="#0d1117", fg=TEXT_COLOR,
                         insertbackground=TEXT_COLOR,
                         font=("Segoe UI", 10), relief=tk.FLAT)
        entry.pack(padx=14, pady=4)
        entry.focus_set()
        entry.select_range(0, tk.END)
        entry.bind("<Return>", lambda e: dlg.destroy())
        tk.Button(dlg, text="OK", command=dlg.destroy,
                  bg=ACCENT_COLOR, fg="#000", relief=tk.FLAT,
                  padx=20, pady=4).pack(pady=8)
        dlg.transient(self.root)
        dlg.grab_set()
        self.root.wait_window(dlg)
        return var.get().strip()

    def _status(self, msg: str):
        self._status_var.set(msg)
        self.root.update_idletasks()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PrepByEKS -- TAPIR Point Tracker v2")
    parser.add_argument("--video", type=str, default=None)
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1400x860")
    TAPIRApp(root, initial_video=args.video)
    root.mainloop()


if __name__ == "__main__":
    main()
