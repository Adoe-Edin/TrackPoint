# identify_popup_tapir.py -- Modal identify popup for Tapir-Tracker
#
# Usage:
#   from identify_popup_tapir import show_identify_popup
#   show_identify_popup(root, face_manager, bgr_frame, app_root=APP_ROOT)

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Optional
import threading

BG=  "#0b0f1a"; PANEL="#111827"; PANEL2="#0d1520"; BORDER="#1e2d45"
ACCENT="#00c9a7"; WARN="#f59e0b"; GOOD="#22c55e"; TEXT="#e2e8f0"
MUTED="#64748b"; TEXT_D="#94a3b8"


def show_identify_popup(parent, face_manager, bgr_frame, app_root=None):
    popup = IdentifyPopup(parent, face_manager, bgr_frame, app_root)
    parent.wait_window(popup.win)
    return popup.result


class IdentifyPopup:
    def __init__(self, parent, face_manager, bgr_frame, app_root):
        self.mgr = face_manager
        self.frame = bgr_frame
        self.app_root = Path(app_root) if app_root else Path(".")
        self.result = None
        self._matches = []

        win = tk.Toplevel(parent)
        self.win = win
        win.title("Identify Athlete -- Face ID")
        win.geometry("680x460")
        win.configure(bg=BG)
        win.resizable(False, False)
        win.grab_set(); win.transient(parent)

        hdr = tk.Frame(win, bg=PANEL, pady=10, padx=16)
        hdr.pack(fill="x")
        tk.Label(hdr, text="[>]  Athlete Identification",
                 bg=PANEL, fg=ACCENT, font=("Consolas", 13, "bold")).pack(side="left")
        self._status_var = tk.StringVar(value="Analysing frame...")
        tk.Label(hdr, textvariable=self._status_var,
                 bg=PANEL, fg=TEXT_D, font=("Consolas", 9)).pack(side="right")

        thumb_row = tk.Frame(win, bg=BG, pady=8); thumb_row.pack(fill="x", padx=16)
        self._thumb_canvas = tk.Canvas(thumb_row, width=160, height=90, bg=PANEL2,
                                        highlightthickness=1, highlightbackground=BORDER)
        self._thumb_canvas.pack(side="left", padx=(0, 16))
        self._thumb_img = None

        thr_frame = tk.Frame(thumb_row, bg=BG); thr_frame.pack(side="left", fill="y")
        tk.Label(thr_frame, text="Threshold:", bg=BG, fg=MUTED,
                 font=("Consolas", 9)).pack(anchor="w")
        self._thr_var = tk.DoubleVar(value=face_manager.threshold)
        tk.Scale(thr_frame, from_=0.40, to=0.95, resolution=0.01, orient="horizontal",
                 variable=self._thr_var, bg=BG, fg=TEXT, troughcolor=BORDER,
                 highlightthickness=0, length=180,
                 command=self._on_thr_change).pack()
        self._thr_lbl = tk.Label(thr_frame, text=f"{face_manager.threshold:.2f}",
                                   bg=BG, fg=ACCENT, font=("Consolas", 10))
        self._thr_lbl.pack(anchor="w")
        tk.Button(thr_frame, text="Re-run", command=self._rerun,
                   bg=BORDER, fg=TEXT, font=("Consolas", 9),
                   relief="flat", padx=10, pady=4, cursor="hand2").pack(anchor="w", pady=(6,0))

        tv_frame = tk.Frame(win, bg=BG)
        tv_frame.pack(fill="both", expand=True, padx=16, pady=(0, 6))
        tk.Label(tv_frame, text="Top-5 matches -- click a row, then Confirm:",
                 bg=BG, fg=MUTED, font=("Consolas", 9)).pack(anchor="w", pady=(0, 4))
        cols = ("#", "Name", "Sport", "Team", "Similarity", "Status")
        tv = ttk.Treeview(tv_frame, columns=cols, show="headings", height=7, selectmode="browse")
        for c in cols: tv.heading(c, text=c)
        tv.column("#", width=30, stretch=False); tv.column("Name", width=160)
        tv.column("Sport", width=110); tv.column("Team", width=110)
        tv.column("Similarity", width=90, stretch=False); tv.column("Status", width=80, stretch=False)
        tv.pack(fill="both", expand=True)
        tv.tag_configure("match", foreground=GOOD); tv.tag_configure("low", foreground=WARN)
        tv.bind("<<TreeviewSelect>>", self._on_select)
        self._tv = tv

        conf = tk.Frame(win, bg=PANEL, pady=6, padx=16); conf.pack(fill="x")
        tk.Label(conf, text="Selected:", bg=PANEL, fg=MUTED, font=("Consolas", 9)).pack(side="left")
        self._sel_var = tk.StringVar(value="--")
        tk.Label(conf, textvariable=self._sel_var, bg=PANEL, fg=ACCENT,
                 font=("Consolas", 11, "bold")).pack(side="left", padx=10)

        btn_row = tk.Frame(win, bg=BG, pady=8); btn_row.pack(fill="x", padx=16)
        self._confirm_btn = tk.Button(btn_row, text="Confirm",
            command=self._confirm, bg=GOOD, fg="#000", font=("Consolas", 11, "bold"),
            relief="flat", padx=18, pady=7, state="disabled", cursor="hand2")
        self._confirm_btn.pack(side="left", padx=(0, 10))
        tk.Button(btn_row, text="Cancel", command=win.destroy,
                  bg=BORDER, fg=TEXT, font=("Consolas", 10),
                  relief="flat", padx=14, pady=7, cursor="hand2").pack(side="left")
        self._path_var = tk.StringVar(value="")
        tk.Label(btn_row, textvariable=self._path_var, bg=BG, fg=TEXT_D,
                 font=("Consolas", 8), wraplength=340, anchor="w").pack(side="left", padx=14)

        self._draw_thumbnail()
        win.after(80, self._run_identify)

    def _draw_thumbnail(self):
        if self.frame is None:
            self._thumb_canvas.create_text(80, 45, text="No frame",
                                            fill=MUTED, font=("Consolas", 9)); return
        try:
            from PIL import Image, ImageTk
            import cv2
            h, w = self.frame.shape[:2]; scale = min(160/w, 90/h)
            nw, nh = int(w*scale), int(h*scale)
            rgb = cv2.cvtColor(cv2.resize(self.frame, (nw, nh)), cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._thumb_canvas.create_image(80, 45, anchor="center", image=img)
            self._thumb_img = img
        except Exception:
            self._thumb_canvas.create_text(80, 45, text="Preview N/A",
                                            fill=MUTED, font=("Consolas", 9))

    def _run_identify(self):
        self._status_var.set("Running face detection...")
        self._tv.delete(*self._tv.get_children())
        if self.frame is None:
            self._status_var.set("Warning: No frame -- load a video first"); return
        threading.Thread(target=self._identify_worker, daemon=True).start()

    def _identify_worker(self):
        thr = self._thr_var.get()
        matches = self.mgr.identify(self.frame.copy(), top_k=5, threshold=thr)
        self.win.after(0, lambda: self._populate(matches))

    def _populate(self, matches):
        self._matches = matches; tv = self._tv; tv.delete(*tv.get_children())
        if not matches:
            self._status_var.set("Warning: No face detected in this frame"); return
        best = matches[0]; above = sum(1 for m in matches if m.above_threshold)
        self._status_var.set(
            f"{'OK' if best.above_threshold else 'Low'}  Best: {best.name}  "
            f"sim={best.similarity:.3f}  |  {above}/{len(matches)} above threshold")
        for m in matches:
            tv.insert("", "end", iid=str(m.rank), values=(
                m.rank, m.name, m.sport, m.team, f"{m.similarity:.4f}",
                "Match" if m.above_threshold else "Low"),
                tags=("match" if m.above_threshold else "low",))
        if best.above_threshold:
            tv.selection_set(str(best.rank)); self._preview(best)

    def _on_select(self, _=None):
        sel = self._tv.selection()
        if not sel: return
        match = next((m for m in self._matches if m.rank == int(sel[0])), None)
        if match: self._preview(match)

    def _preview(self, match):
        abs_path = self.app_root / "Tapir-Tracker" / match.session_path
        self._sel_var.set(f"  {match.name}   |   {match.sport}   |   sim {match.similarity:.3f}")
        self._path_var.set(f"->  {abs_path}")
        self._confirm_btn.config(state="normal")

    def _confirm(self):
        sel = self._tv.selection()
        if not sel: return
        match = next((m for m in self._matches if m.rank == int(sel[0])), None)
        if not match: return
        try:
            from face_id_tab_tapir import athlete_store
            athlete_store.set(match)
        except Exception:
            pass
        abs_path = self.app_root / "Tapir-Tracker" / match.session_path
        abs_path.mkdir(parents=True, exist_ok=True)
        self.result = match; self.win.destroy()

    def _on_thr_change(self, val):
        self._thr_lbl.config(text=f"{float(val):.2f}")

    def _rerun(self):
        self._run_identify()