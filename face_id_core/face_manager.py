"""
face_manager.py
===============
High-level orchestrator: webcam scan protocol → enrolment → identification.

Single entry point consumed by all apps (Tracker-RTM, Tracker-Pose, Tapir, Movement-Counter).

Usage
-----
    mgr = FaceManager()                           # InsightFace default
    mgr.set_backend("yunet_sface")                # switch any time

    # Enrolment
    mgr.start_enrolment()
    for frame in webcam_frames:                   # 5 frames, ~700 ms apart
        mgr.scan_frame(frame)
    athlete_id = mgr.finish_enrolment("Name", "Sport")

    # Identification
    matches = mgr.identify(frame)                 # list[MatchResult], top-5
    best    = matches[0]
    if best.above_threshold:
        print(best.name, best.session_path)
"""

from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from .embedding_engine import EmbeddingEngine
from .athlete_db       import AthleteDB, APP_ROOT

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.65
SCAN_FRAMES       = 5
SCAN_ANGLES       = ["Front", "Left 30°", "Right 30°", "Up 15°", "Down 15°"]


@dataclass
class ScanResult:
    frame_index: int
    angle_label: str
    embedding:   Optional[np.ndarray]
    success:     bool


@dataclass
class MatchResult:
    athlete_id:      str
    name:            str
    sport:           str
    team:            str
    similarity:      float
    above_threshold: bool
    session_path:    str
    rank:            int = 1


@dataclass
class EnrolmentState:
    frames_done: int   = 0
    embeddings:  list  = field(default_factory=list)
    complete:    bool  = False


class FaceManager:
    """
    Central face recognition manager (thread-safe).

    Parameters
    ----------
    backend    : "insightface" | "yunet_sface"
    gpu        : use GPU
    db_path    : explicit path to athletes.json (None = auto-resolve)
    threshold  : cosine similarity threshold (default 0.65, user-adjustable)
    on_status  : callback(str) — receives status messages for the UI status bar
    """

    def __init__(
        self,
        backend:   str                  = "insightface",
        gpu:       bool                 = True,
        db_path:   Optional[Path | str] = None,
        threshold: float                = DEFAULT_THRESHOLD,
        on_status: Optional[Callable]   = None,
    ):
        self.threshold  = threshold
        self._on_status = on_status or (lambda msg: None)
        self._lock      = threading.Lock()
        self._enrol     = EnrolmentState()

        self._status("Initialising Face-ID engine…")
        self.engine = EmbeddingEngine(backend=backend, gpu=gpu)
        self.db     = AthleteDB(db_path=db_path)
        self._status(
            f"Face-ID ready — {self.engine.backend_label()}  ·  "
            f"{self.db.count()} athlete(s) enrolled  ·  DB: {self.db.db_location}"
        )

    # ── backend switching ─────────────────────────────────────────────────────

    def set_backend(self, backend: str, gpu: bool = True) -> bool:
        """Switch embedding backend at runtime.  Returns True on success."""
        self._status(f"Switching to {backend}…")
        try:
            new = EmbeddingEngine(backend=backend, gpu=gpu)
            if new.is_ready():
                with self._lock:
                    self.engine = new
                self._status(f"Backend → {new.backend_label()}")
                return True
            self._status("Backend switch failed — model unavailable")
            return False
        except Exception as exc:
            self._status(f"Backend switch error: {exc}")
            return False

    # ── enrolment ─────────────────────────────────────────────────────────────

    def start_enrolment(self):
        with self._lock:
            self._enrol = EnrolmentState()
        self._status(f"Enrolment started — {SCAN_FRAMES} frames required")

    def scan_frame(self, bgr_frame: np.ndarray) -> ScanResult:
        """
        Process one webcam frame during the enrolment scan protocol.
        Frame is immediately discarded after embedding extraction.
        Call once per angle (SCAN_FRAMES total calls).
        """
        with self._lock:
            idx = self._enrol.frames_done
            if idx >= SCAN_FRAMES:
                return ScanResult(idx, "—", None, False)

        angle = SCAN_ANGLES[idx] if idx < len(SCAN_ANGLES) else f"Frame {idx+1}"
        self._status(f"Capturing {angle}  ({idx+1}/{SCAN_FRAMES})…")

        emb     = self.engine.extract(bgr_frame)   # frame discarded here
        success = emb is not None

        with self._lock:
            if success:
                self._enrol.embeddings.append(emb)
            self._enrol.frames_done += 1
            if self._enrol.frames_done >= SCAN_FRAMES:
                self._enrol.complete = True

        self._status(
            f"{'✓' if success else '✗'} {angle} — "
            f"{'face detected' if success else 'no face — retry'}"
        )
        return ScanResult(idx, angle, emb, success)

    def finish_enrolment(
        self,
        name:  str,
        sport: str,
        team:  str = "",
        notes: str = "",
    ) -> Optional[str]:
        """
        Average collected embeddings, save profile, return athlete_id.
        Returns None if fewer than 2 usable frames were captured.
        """
        with self._lock:
            embeddings = list(self._enrol.embeddings)

        if len(embeddings) < 2:
            self._status("⚠ Not enough face captures — please re-scan")
            return None

        master = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
        norm   = np.linalg.norm(master)
        if norm > 1e-8:
            master /= norm

        athlete_id = self.db.enroll(name, sport, master, team, notes)
        self._status(
            f"✅ Enrolled: {name}  ({sport})  ·  "
            f"{len(embeddings)}/{SCAN_FRAMES} frames used"
        )
        with self._lock:
            self._enrol = EnrolmentState()
        return athlete_id

    # ── identification ────────────────────────────────────────────────────────

    def identify(
        self,
        bgr_frame: np.ndarray,
        top_k:     int            = 5,
        threshold: Optional[float] = None,
    ) -> list[MatchResult]:
        """
        Identify the athlete in bgr_frame.
        Returns top-k MatchResult objects sorted by similarity (best first).
        Frame is not stored.
        """
        th  = threshold if threshold is not None else self.threshold
        emb = self.engine.extract(bgr_frame)
        if emb is None:
            self._status("No face detected in frame")
            return []
        return self._match(emb, th, top_k)

    def identify_from_embedding(
        self,
        embedding: np.ndarray,
        top_k:     int            = 5,
        threshold: Optional[float] = None,
    ) -> list[MatchResult]:
        """Identify from a pre-extracted embedding (avoids re-running the model)."""
        th = threshold if threshold is not None else self.threshold
        return self._match(embedding, th, top_k)

    def _match(self, emb: np.ndarray, th: float, top_k: int) -> list[MatchResult]:
        raw = self.db.identify(emb, threshold=th, top_k=top_k)
        results = [
            MatchResult(
                athlete_id=r["athlete_id"],
                name=r["name"],
                sport=r["sport"],
                team=r.get("team", ""),
                similarity=r["similarity"],
                above_threshold=r["above_threshold"],
                session_path=r["session_path"],
                rank=i + 1,
            )
            for i, r in enumerate(raw)
        ]
        if results:
            best = results[0]
            if best.above_threshold:
                self._status(f"✅ Identified: {best.name}  (sim={best.similarity:.3f})")
            else:
                self._status(
                    f"⚠ Best match: {best.name}  (sim={best.similarity:.3f})  "
                    f"— below threshold {th:.2f}"
                )
        return results

    # ── properties / helpers ──────────────────────────────────────────────────

    @property
    def enrolled_count(self) -> int:
        return self.db.count()

    @property
    def enrolment_progress(self) -> tuple[int, int]:
        with self._lock:
            return self._enrol.frames_done, SCAN_FRAMES

    @property
    def enrolment_complete(self) -> bool:
        with self._lock:
            return self._enrol.complete

    def set_threshold(self, value: float):
        import numpy as _np
        self.threshold = float(_np.clip(value, 0.0, 1.0))

    def all_athletes(self) -> list[dict]:
        return self.db.all_athletes()

    def delete_athlete(self, athlete_id: str) -> bool:
        return self.db.delete(athlete_id)

    def db_location(self) -> str:
        return self.db.db_location

    def _status(self, msg: str):
        log.info(msg)
        try:
            self._on_status(msg)
        except Exception:
            pass
