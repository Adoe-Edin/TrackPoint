"""
athlete_db.py
=============
JSON-backed athlete identity database for ISN PrepByEKS.

DB path resolution (first match wins):
  1. Explicit path passed to AthleteDB(db_path=...)
  2. ISN_FACE_DB environment variable  ← point at cloud/NAS sync folder
  3. <APP_ROOT>/data/athletes.json     ← local fallback

APP_ROOT = C:\\Users\\ISN\\Documents\\Personal\\Consult\\App\\
           (shared parent of Tracker-RTM, Tracker-Pose, Tapir, Movement-Counter)

To use cloud/NAS sync in PowerShell:
    $Env:ISN_FACE_DB = "C:\\Users\\ISN\\OneDrive\\ISN\\data\\athletes.json"

Privacy guarantee
-----------------
- Only stores: name, sport, team, notes, created_at, updated_at, embedding vector.
- NEVER stores images, video frames, or any pixel data.
- Embedding vectors are float arrays — they cannot be reversed to reconstruct a face.
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from .embedding_engine import cosine_similarity

log = logging.getLogger(__name__)

# ── APP_ROOT: shared parent for all ISN apps ──────────────────────────────────
# Tracker-RTM lives at  APP_ROOT / Tracker-RTM /
# This file lives at    APP_ROOT / Tracker-RTM / backend / app / face_id_core /
APP_ROOT     = Path(__file__).resolve().parents[4]   # .../App/
_DEFAULT_DB  = APP_ROOT / "data" / "athletes.json"


class AthleteDB:
    """
    Persistent athlete identity store.

    Parameters
    ----------
    db_path : str | Path | None
        Explicit path to athletes.json.  None = auto-resolve.
    """

    def __init__(self, db_path: Optional[Path | str] = None):
        if db_path:
            self.db_path = Path(db_path)
        elif "ISN_FACE_DB" in os.environ:
            self.db_path = Path(os.environ["ISN_FACE_DB"])
        else:
            self.db_path = _DEFAULT_DB

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def enroll(
        self,
        name:      str,
        sport:     str,
        embedding: np.ndarray,
        team:      str = "",
        notes:     str = "",
    ) -> str:
        """
        Add or update an athlete profile.
        Returns the athlete_id (slug).
        No image data is stored — only the embedding vector.
        """
        athlete_id = self._make_id(name, sport)
        now = datetime.utcnow().isoformat()

        for rec in self._records:
            if rec["athlete_id"] == athlete_id:
                rec["embedding"]  = embedding.tolist()
                rec["updated_at"] = now
                rec["team"]       = team  or rec.get("team", "")
                rec["notes"]      = notes or rec.get("notes", "")
                self._save()
                log.info("Updated profile: %s", athlete_id)
                return athlete_id

        self._records.append({
            "athlete_id": athlete_id,
            "name":       name.strip(),
            "sport":      sport.strip(),
            "team":       team.strip(),
            "notes":      notes.strip(),
            "embedding":  embedding.tolist(),
            "created_at": now,
            "updated_at": now,
        })
        self._save()
        log.info("Enrolled new athlete: %s", athlete_id)
        return athlete_id

    def identify(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.65,
        top_k:     int   = 5,
    ) -> list[dict]:
        """
        Find the top-k closest athletes to query_embedding.

        Returns list of dicts sorted by similarity (descending):
            name, sport, team, similarity, athlete_id,
            session_path, above_threshold
        """
        if not self._records:
            return []

        q      = np.asarray(query_embedding, dtype=np.float32)
        scored = []
        for rec in self._records:
            emb = np.asarray(rec["embedding"], dtype=np.float32)
            sim = cosine_similarity(q, emb)
            scored.append({
                "athlete_id":      rec["athlete_id"],
                "name":            rec["name"],
                "sport":           rec["sport"],
                "team":            rec.get("team", ""),
                "similarity":      round(float(sim), 4),
                "session_path":    self._suggest_session_path(rec),
                "above_threshold": sim >= threshold,
            })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def all_athletes(self) -> list[dict]:
        """Return all enrolled athletes (without embedding vectors)."""
        return [
            {k: v for k, v in rec.items() if k != "embedding"}
            for rec in self._records
        ]

    def delete(self, athlete_id: str) -> bool:
        before = len(self._records)
        self._records = [r for r in self._records if r["athlete_id"] != athlete_id]
        if len(self._records) < before:
            self._save()
            return True
        return False

    def count(self) -> int:
        return len(self._records)

    @property
    def db_location(self) -> str:
        return str(self.db_path)

    # ── session path ──────────────────────────────────────────────────────────

    @staticmethod
    def _suggest_session_path(rec: dict) -> str:
        """
        Suggest a session folder relative to APP_ROOT/Tracker-RTM/sessions/:
            sessions/<sport>/<first_name>/<YYYY-MM-DD>/
        """
        sport = rec["sport"].lower().replace(" ", "_")
        first = rec["name"].split()[0].lower()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return f"sessions/{sport}/{first}/{today}/"

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._records = data.get("athletes", [])
                log.info("Loaded %d athlete(s) from %s", len(self._records), self.db_path)
            except Exception as exc:
                log.error("Failed to load athlete DB: %s", exc)
                self._records = []
        else:
            self._records = []

    def _save(self):
        payload = {
            "schema_version": "1.0",
            "updated_at":     datetime.utcnow().isoformat(),
            "app_root":       str(APP_ROOT),
            "athletes":       self._records,
        }
        tmp = self.db_path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp.replace(self.db_path)   # atomic on same filesystem
        except Exception as exc:
            log.error("Failed to save athlete DB: %s", exc)

    @staticmethod
    def _make_id(name: str, sport: str) -> str:
        return (
            f"{name.strip().lower().replace(' ', '_')}"
            f"__{sport.strip().lower().replace(' ', '_')}"
        )
