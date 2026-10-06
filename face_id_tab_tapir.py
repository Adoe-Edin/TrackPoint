# face_id_tab_tapir.py -- Face-ID tab for Tapir-Tracker
#
# Integration in main.py (3 steps):
#   1. from face_id_core import FaceManager
#      from face_id_tab_tapir import FaceIDTab, make_name_label, athlete_store, grab_frame_tapir
#      APP_ROOT = Path(__file__).parent.parent
#   2. self.face_mgr = FaceManager(on_status=self._status)
#   3. In _build_ui after the last nb.add():
#      self._faceid_tab_frame = tk.Frame(self.notebook, bg=BG_COLOR)
#      self.notebook.add(self._faceid_tab_frame, text="  [ID] Face ID  ")
#      self._face_id_tab = FaceIDTab(self._faceid_tab_frame, manager=self.face_mgr, app_root=APP_ROOT)
#      And add the name-ID bar to each canvas tab builder -- see make_name_label()

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading, time, logging, queue
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger(__name__)

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

from face_id_core import FaceManager, SCAN_FRAMES, SCAN_ANGLES

# theme -- matches Tapir dark palette
BG     = "#0b0f1a"
PANEL  = "#111827"
PANEL2 = "#0d1520"
BORDER = "#1e2d45"
ACCENT = "#00c9a7"
WARN   = "#f59e0b"
DANGER = "#ef4444"
GOOD   = "#22c55e"
TEXT   = "#e2e8f0"
MUTED  = "#64748b"
TEXT_D = "#94a3b8"
FONT_MONO = ("Consolas", 10)
FONT_LG   = ("Consolas", 12, "bold")


# -- singleton athlete store --------------------------------------------------
class ActiveAthleteStore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._match = None
            cls._instance._cbs   = []
        return cls._instance
    def set(self, match):
        self._match = match
        for cb in self._cbs:
            try: cb(match)
            except Exception: pass
    def get(self): return self._match
    def on_change(self, cb: Callable): self._cbs.append(cb)
    def clear(self): self.set(None)

athlete_store = ActiveAthleteStore()


def make_name_label(parent, store=None, font_size: int = 9) -> tk.Label:
    """Auto-updating name label. Pack at top of any video-bearing tab frame."""
    store = store or athlete_store
    var = tk.StringVar(value="[?]  No athlete identified")
    lbl = tk.Label(parent, textvariable=var, bg=PANEL2, fg=ACCENT,
                   font=("Consolas", font_size, "bold"), anchor="w", padx=10, pady=3)
    def _update(match):
        if match and match.above_threshold:
            var.set(f"[ID]  {match.name}   |   {match.sport}   |   sim {match.similarity:.3f}")
            lbl.config(fg=ACCENT)
        elif match:
            var.set(f"[?]  {match.name} ?   |   sim {match.similarity:.3f}  (below threshold)")
            lbl.config(fg=WARN)
        else:
            var.set("[?]  No athlete identified")
            lbl.config(fg=MUTED)
    store.on_change(_update)
    return lbl


def session_abs_path(app_root: Path, session_rel: str) -> Path:
    return app_root / "Tapir-Tracker" / session_rel


# -- FaceIDTab ----------------------------------------------------------------
class FaceIDTab:
    _PANELS = ["[?] Profiles", "[+] Enrol", "[>] Identify"]

    def __init__(self, parent, manager: FaceManager, app_root: Path):
        self.mgr = manager
        self.app_root = Path(app_root)
        self._cap = None
        self._cam_running = False
        self._cam_thread  = None
        self._frame_q     = queue.Queue(maxsize=2)
        self._last_frame  = None
        self._collecting  = False
        self._current_matches: list = []
        self.frame = parent
        self._build_ui()
        self._refresh_profiles()

    def _build_ui(self):
        top = tk.Frame(self.frame, bg=PANEL, pady=6, padx=10)
        top.pack(fill="x", side="top")
        self._status_var = tk.StringVar(value="Initialising...")
        tk.Label(top, textvariable=self._status_var, bg=PANEL, fg=ACCENT,
                 font=FONT_MONO, anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(top, text="Backend:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 9)).pack(side="left", padx=(12, 4))
        self._backend_var = tk.StringVar(value="yunet_sface")
        cb = ttk.Combobox(top, textvariable=self._backend_var,
                          values=["insightface", "yunet_sface"], width=16, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._on_backend_change)
        tk.Label(top, text="  Threshold:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 9)).pack(side="left", padx=(14, 4))
        self._thr_var = tk.DoubleVar(value=self.mgr.threshold)
        tk.Scale(top, from_=0.40, to=0.95, resolution=0.01, orient="horizontal",
                 variable=self._thr_var, bg=PANEL, fg=TEXT, troughcolor=BORDER,
                 highlightthickness=0, length=130,
                 command=self._on_threshold_change).pack(side="left")
        self._thr_lbl = tk.Label(top, text=f"{self.mgr.threshold:.2f}",
                                  bg=PANEL, fg=ACCENT, font=FONT_MONO, width=4)
        self._thr_lbl.pack(side="left", padx=(2, 0))
        sel = tk.Frame(self.frame, bg=BG, pady=6)
        sel.pack(fill="x", padx=10)
        self._panel_btns = []
        for p in self._PANELS:
            b = tk.Button(sel, text=p, command=lambda x=p: self._show_panel(x),
                          bg=BORDER, fg=TEXT_D, relief="flat",
                          font=("Consolas", 10), padx=14, pady=6, cursor="hand2")
            b.pack(side="left", padx=3)
            self._panel_btns.append((p, b))
        self._panels = {}
        self._cont = tk.Frame(self.frame, bg=BG)
        self._cont.pack(fill="both", expand=True, padx=10, pady=6)
        self._build_profiles_panel()
        self._build_enrol_panel()
        self._build_identify_panel()
        self._show_panel(self._PANELS[0])
        self.mgr._on_status = self._set_status

    def _build_profiles_panel(self):
        f = tk.Frame(self._cont, bg=BG)
        self._panels[self._PANELS[0]] = f
        db_row = tk.Frame(f, bg=PANEL, pady=6, padx=10)
        db_row.pack(fill="x", pady=(0, 8))
        tk.Label(db_row, text="Athlete DB:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 9)).pack(side="left")
        self._db_path_var = tk.StringVar(value=self.mgr.db_location())
        tk.Label(db_row, textvariable=self._db_path_var, bg=PANEL, fg=ACCENT,
                 font=("Consolas", 9), wraplength=700, anchor="w").pack(
                 side="left", padx=8, fill="x", expand=True)
        tk.Button(db_row, text="Change...", command=self._change_db_path,
                  bg=BORDER, fg=TEXT, font=("Consolas", 9),
                  relief="flat", padx=10, pady=2).pack(side="right")
        cols = ("Name", "Sport", "Team", "Enrolled", "Session path")
        tv = ttk.Treeview(f, columns=cols, show="headings", height=18)
        for c in cols: tv.heading(c, text=c)
        tv.column("Name", width=170); tv.column("Sport", width=140)
        tv.column("Team", width=130); tv.column("Enrolled", width=100)
        tv.column("Session path", width=300)
        tv.pack(fill="both", expand=True, pady=(0, 6))
        self._profiles_tv = tv
        btn_row = tk.Frame(f, bg=BG)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="Refresh", command=self._refresh_profiles,
                  bg=BORDER, fg=TEXT, relief="flat",
                  font=("Consolas", 9), padx=10, pady=4).pack(side="left", padx=3)
        tk.Button(btn_row, text="Delete selected", command=self._delete_selected,
                  bg=DANGER, fg="#fff", relief="flat",
                  font=("Consolas", 9), padx=10, pady=4).pack(side="left", padx=3)
        self._count_var = tk.StringVar(value="")
        tk.Label(btn_row, textvariable=self._count_var, bg=BG, fg=MUTED,
                 font=("Consolas", 9)).pack(side="right", padx=8)

    def _refresh_profiles(self):
        tv = self._profiles_tv; tv.delete(*tv.get_children())
        for a in self.mgr.all_athletes():
            sport = a["sport"].lower().replace(" ", "_")
            first = a["name"].split()[0].lower()
            tv.insert("", "end", iid=a["athlete_id"], values=(
                a["name"], a["sport"], a.get("team", ""),
                a.get("created_at", "")[:10], f"sessions/{sport}/{first}/<date>/"))
        n = self.mgr.enrolled_count
        self._count_var.set(f"{n} athlete{'s' if n != 1 else ''} enrolled  |  vectors only -- no photos")

    def _delete_selected(self):
        sel = self._profiles_tv.selection()
        if not sel: return
        if messagebox.askyesno("Delete", f"Delete {len(sel)} profile(s)?  Cannot be undone."):
            for aid in sel: self.mgr.delete_athlete(aid)
            self._refresh_profiles()

    def _change_db_path(self):
        p = filedialog.askdirectory(title="Select database folder")
        if p:
            new_db = Path(p) / "athletes.json"
            self.mgr.db.db_path = new_db; self.mgr.db._load()
            self._db_path_var.set(str(new_db)); self._refresh_profiles()
            self._set_status(f"Database -> {new_db}")
    def _build_enrol_panel(self):
        f = tk.Frame(self._cont, bg=BG)
        self._panels[self._PANELS[1]] = f
        left = tk.Frame(f, bg=BG); right = tk.Frame(f, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right.pack(side="left", fill="y")
        self._enrol_name  = self._field(left, "ATHLETE NAME *")
        self._enrol_sport = self._field(left, "SPORT / DISCIPLINE *")
        self._enrol_team  = self._field(left, "TEAM / CLUB")
        self._enrol_notes = self._field(left, "NOTES")
        dots_row = tk.Frame(left, bg=BG); dots_row.pack(pady=12)
        tk.Label(dots_row, text="Scan progress:  ", bg=BG, fg=MUTED,
                 font=FONT_MONO).pack(side="left")
        self._scan_dots = []
        for _ in SCAN_ANGLES:
            c = tk.Canvas(dots_row, width=18, height=18, bg=BG, highlightthickness=0)
            c.create_oval(2, 2, 16, 16, fill=BORDER, outline="", tags="dot")
            c.pack(side="left", padx=3)
            self._scan_dots.append(c)
        self._angle_var = tk.StringVar(value="")
        tk.Label(left, textvariable=self._angle_var, bg=BG, fg=ACCENT,
                 font=FONT_MONO).pack(anchor="w")
        tk.Label(left, text="Privacy: 5 angles captured | frame -> vector -> deleted. No photos.",
                 bg=BG, fg=MUTED, font=("Consolas", 8), justify="left").pack(anchor="w", pady=(8, 4))
        btn_row = tk.Frame(left, bg=BG); btn_row.pack(fill="x", pady=6)
        self._scan_btn = tk.Button(btn_row, text=">> Start Scan",
            command=self._start_enrolment, bg=ACCENT, fg="#000", font=FONT_LG,
            relief="flat", padx=16, pady=8, cursor="hand2")
        self._scan_btn.pack(side="left", padx=(0, 8))
        self._save_btn = tk.Button(btn_row, text="Save Profile",
            command=self._finish_enrolment, bg=GOOD, fg="#000", font=FONT_LG,
            relief="flat", padx=16, pady=8, state="disabled", cursor="hand2")
        self._save_btn.pack(side="left")
        tk.Label(right, text="Webcam preview", bg=BG, fg=MUTED,
                 font=("Consolas", 8)).pack()
        self._enrol_canvas = tk.Canvas(right, width=320, height=240, bg=PANEL2,
                                        highlightthickness=1, highlightbackground=BORDER)
        self._enrol_canvas.pack()
        self._enrol_canvas_img = None

    def _start_enrolment(self):
        self.mgr.start_enrolment()
        self._scan_btn.config(state="disabled", text="Scanning...")
        self._save_btn.config(state="disabled")
        self._reset_dots(); self._angle_var.set("")
        self._ensure_camera(); self._collecting = True; self._enrol_tick()

    def _enrol_tick(self):
        if not self._collecting: return
        frame = self._last_frame
        if frame is not None:
            result = self.mgr.scan_frame(frame.copy())
            self._update_dot(result.frame_index, result.success)
            self._angle_var.set(
                f"{'OK' if result.success else 'X'}  {result.angle_label}  "
                f"({result.frame_index+1}/{SCAN_FRAMES})")
            if self.mgr.enrolment_complete:
                self._collecting = False
                good = sum(1 for c in self._scan_dots if c.itemcget("dot", "fill") == GOOD)
                if good >= 2:
                    self._save_btn.config(state="normal")
                    self._scan_btn.config(state="normal", text="Re-scan")
                    self._angle_var.set(f"OK  {good}/{SCAN_FRAMES} captures -- ready to save")
                else:
                    self._scan_btn.config(state="normal", text=">> Start Scan")
                    self._angle_var.set("Warning: Too few faces -- re-scan")
                return
        self.frame.after(700, self._enrol_tick)

    def _finish_enrolment(self):
        name = self._enrol_name.get().strip()
        sport = self._enrol_sport.get().strip()
        if not name or not sport:
            messagebox.showwarning("Missing fields", "Name and Sport are required.")
            return
        aid = self.mgr.finish_enrolment(name, sport,
            team=self._enrol_team.get().strip(), notes=self._enrol_notes.get().strip())
        if aid:
            messagebox.showinfo("Enrolled", f"OK  {name} enrolled successfully!")
            for v in (self._enrol_name, self._enrol_sport, self._enrol_team, self._enrol_notes):
                v.set("")
            self._reset_dots(); self._angle_var.set("")
            self._save_btn.config(state="disabled")
            self._scan_btn.config(state="normal", text=">> Start Scan")
            self._refresh_profiles(); self._show_panel(self._PANELS[0])

    def _reset_dots(self):
        for c in self._scan_dots: c.itemconfig("dot", fill=BORDER)

    def _update_dot(self, idx, success):
        if 0 <= idx < len(self._scan_dots):
            self._scan_dots[idx].itemconfig("dot", fill=GOOD if success else DANGER)
    def _build_identify_panel(self):
        f = tk.Frame(self._cont, bg=BG)
        self._panels[self._PANELS[2]] = f
        left = tk.Frame(f, bg=BG); right = tk.Frame(f, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right.pack(side="left", fill="y")
        tk.Label(left, text="Match Shortlist (top 5 -- ranked by similarity)",
                 bg=BG, fg=MUTED, font=("Consolas", 9)).pack(anchor="w", pady=(0, 4))
        cols = ("#", "Name", "Sport", "Team", "Similarity", "Status")
        tv = ttk.Treeview(left, columns=cols, show="headings", height=7)
        for c in cols: tv.heading(c, text=c)
        tv.column("#", width=30); tv.column("Name", width=160)
        tv.column("Sport", width=120); tv.column("Team", width=120)
        tv.column("Similarity", width=90); tv.column("Status", width=90)
        tv.pack(fill="x", pady=(0, 4))
        self._match_tv = tv
        tv.tag_configure("match", foreground=GOOD)
        tv.tag_configure("low",   foreground=WARN)
        tv.bind("<<TreeviewSelect>>", self._on_shortlist_select)
        conf = tk.Frame(left, bg=PANEL, pady=8, padx=12)
        conf.pack(fill="x", pady=8)
        tk.Label(conf, text="Confirmed athlete:", bg=PANEL, fg=MUTED,
                 font=("Consolas", 9)).pack(anchor="w")
        self._conf_name_var = tk.StringVar(value="--")
        tk.Label(conf, textvariable=self._conf_name_var, bg=PANEL, fg=ACCENT,
                 font=("Consolas", 14, "bold")).pack(anchor="w")
        self._conf_path_var = tk.StringVar(value="")
        tk.Label(conf, textvariable=self._conf_path_var, bg=PANEL, fg=TEXT_D,
                 font=("Consolas", 9)).pack(anchor="w")
        act_row = tk.Frame(left, bg=BG); act_row.pack(fill="x", pady=4)
        tk.Button(act_row, text="Confirm selected", command=self._confirm_selected,
                  bg=GOOD, fg="#000", font=FONT_LG, relief="flat",
                  padx=12, pady=6, cursor="hand2").pack(side="left", padx=(0, 8))
        tk.Button(act_row, text="Copy path", command=self._copy_path,
                  bg=BORDER, fg=TEXT, font=("Consolas", 10),
                  relief="flat", padx=12, pady=6).pack(side="left")
        tk.Label(right, text="Webcam preview", bg=BG, fg=MUTED,
                 font=("Consolas", 8)).pack()
        self._ident_canvas = tk.Canvas(right, width=320, height=240, bg=PANEL2,
                                        highlightthickness=1, highlightbackground=BORDER)
        self._ident_canvas.pack()
        self._ident_canvas_img = None
        tk.Button(right, text="Identify Athlete", command=self._run_identify,
                  bg=ACCENT, fg="#000", font=FONT_LG,
                  relief="flat", padx=16, pady=10, cursor="hand2").pack(pady=8)
        tk.Label(right, text="Continuous mode:", bg=BG, fg=MUTED,
                 font=("Consolas", 8)).pack()
        self._cont_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right, text="Auto-identify every 2 s", variable=self._cont_var,
                        bg=BG, fg=TEXT_D, selectcolor=PANEL, font=("Consolas", 9),
                        command=self._toggle_continuous).pack()

    def _run_identify(self):
        self._ensure_camera()
        frame = self._last_frame
        if frame is None:
            self._set_status("Warning: No webcam frame available"); return
        matches = self.mgr.identify(frame.copy())
        self._populate_shortlist(matches)

    def _populate_shortlist(self, matches):
        self._current_matches = matches
        tv = self._match_tv; tv.delete(*tv.get_children())
        for m in matches:
            tv.insert("", "end", iid=str(m.rank), values=(
                m.rank, m.name, m.sport, m.team, f"{m.similarity:.4f}",
                "Match" if m.above_threshold else "Low"),
                tags=("match" if m.above_threshold else "low",))
        if matches and matches[0].above_threshold:
            tv.selection_set(str(matches[0].rank))
            self._preview_match(matches[0])

    def _on_shortlist_select(self, _=None):
        sel = self._match_tv.selection()
        if not sel: return
        match = next((m for m in self._current_matches if m.rank == int(sel[0])), None)
        if match: self._preview_match(match)

    def _preview_match(self, match):
        abs_path = session_abs_path(self.app_root, match.session_path)
        self._conf_name_var.set(
            f"  {match.name}   |   {match.sport}   |   sim {match.similarity:.3f}")
        self._conf_path_var.set(f"  {abs_path}")

    def _confirm_selected(self):
        sel = self._match_tv.selection()
        if not sel:
            messagebox.showinfo("Select", "Click a row first."); return
        match = next((m for m in self._current_matches if m.rank == int(sel[0])), None)
        if match: self._apply_match(match)

    def _apply_match(self, match):
        athlete_store.set(match)
        abs_path = session_abs_path(self.app_root, match.session_path)
        abs_path.mkdir(parents=True, exist_ok=True)
        self._conf_name_var.set(
            f"  {match.name}   |   {match.sport}   |   sim {match.similarity:.3f}")
        self._conf_path_var.set(f"  {abs_path}")
        self._set_status(f"Active: {match.name}  |  session: {abs_path}")

    def _copy_path(self):
        val = self._conf_path_var.get().strip()
        if val:
            self.frame.clipboard_clear(); self.frame.clipboard_append(val)
            self._set_status(f"Copied: {val}")

    def _toggle_continuous(self):
        if self._cont_var.get(): self._continuous_loop()

    def _continuous_loop(self):
        if not self._cont_var.get(): return
        self._run_identify(); self.frame.after(2000, self._continuous_loop)

    def _ensure_camera(self):
        if self._cam_running: return
        if not _CV2_OK:
            self._set_status("cv2 not available"); return
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._set_status("Warning: No webcam detected"); return
        self._cam_running = True
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()
        self._update_canvases()

    def _cam_loop(self):
        while self._cam_running:
            ret, frame = self._cap.read()
            if ret:
                self._last_frame = frame
                if not self._frame_q.full(): self._frame_q.put(frame)
            time.sleep(0.033)

    def _update_canvases(self):
        if not self._frame_q.empty():
            frame = self._frame_q.get_nowait()
            self._draw_frame(frame, self._enrol_canvas, "_enrol_canvas_img")
            self._draw_frame(frame, self._ident_canvas, "_ident_canvas_img")
        self.frame.after(33, self._update_canvases)

    def _draw_frame(self, frame, canvas, img_attr):
        if not _PIL_OK: return
        h, w = frame.shape[:2]
        cw, ch = int(canvas["width"]), int(canvas["height"])
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=img)
        setattr(self, img_attr, img)

    def _show_panel(self, name):
        for p, b in self._panel_btns:
            active = (p == name)
            b.config(bg=ACCENT if active else BORDER, fg="#000" if active else TEXT_D,
                     font=("Consolas", 10, "bold" if active else "normal"))
        for k, f in self._panels.items(): f.pack_forget()
        self._panels[name].pack(fill="both", expand=True)
        if name == self._PANELS[2]: self._ensure_camera()

    def _field(self, parent, label):
        tk.Label(parent, text=label, bg=BG, fg=MUTED,
                 font=("Consolas", 9)).pack(anchor="w", pady=(8, 2))
        var = tk.StringVar()
        tk.Entry(parent, textvariable=var, bg=PANEL2, fg=TEXT,
                 insertbackground=ACCENT, relief="flat", font=("Consolas", 11),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT).pack(fill="x")
        return var

    def _set_status(self, msg):
        try: self._status_var.set(msg)
        except Exception: pass

    def _on_backend_change(self, _=None):
        backend = self._backend_var.get()
        self.frame.after(50, lambda: self.mgr.set_backend(backend))

    def _on_threshold_change(self, val):
        v = float(val); self.mgr.set_threshold(v)
        self._thr_lbl.config(text=f"{v:.2f}")

    def stop_camera(self):
        self._cam_running = False
        if self._cap: self._cap.release()


# -- frame grab ---------------------------------------------------------------
def grab_frame_tapir(app):
    """Grab best available BGR frame from Tapir-Tracker."""
    try:
        if app.state.frames:
            idx = max(0, min(app.state.current_frame, len(app.state.frames) - 1))
            return app.state.frames[idx].copy()
    except Exception:
        pass
    try:
        if app._trim_frame_bgr is not None:
            return app._trim_frame_bgr.copy()
    except Exception:
        pass
    return None