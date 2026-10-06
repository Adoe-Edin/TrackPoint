"""
face_id_core — ISN PrepByEKS face recognition package
Shared across: Tracker-RTM, Tracker-Pose, Tapir-Tracker, Movement-Counter

Set env var ISN_FACE_DB to point at your cloud/NAS-synced athletes.json:
    $Env:ISN_FACE_DB = "C:\\OneDrive\\ISN\\data\\athletes.json"
"""
from .embedding_engine import EmbeddingEngine, cosine_similarity
from .athlete_db       import AthleteDB
from .face_manager     import FaceManager, MatchResult, ScanResult, SCAN_FRAMES, SCAN_ANGLES

__all__ = [
    "EmbeddingEngine",
    "cosine_similarity",
    "AthleteDB",
    "FaceManager",
    "MatchResult",
    "ScanResult",
    "SCAN_FRAMES",
    "SCAN_ANGLES",
]
