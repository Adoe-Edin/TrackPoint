"""
embedding_engine.py
====================
Swappable face-embedding backend for ISN PrepByEKS Face-ID system.

Supported backends
------------------
  "insightface"  — InsightFace buffalo_l  (default, ONNX GPU, non-commercial research)
  "yunet_sface"  — YuNet detector + SFace recogniser (Apache 2.0, CPU/GPU, any use)

Switching backend at runtime:
    engine = EmbeddingEngine(backend="yunet_sface")
    # or via FaceManager.set_backend("yunet_sface")

Privacy guarantee
-----------------
  - No frame / pixel data is ever written to disk by this module.
  - Only normalised 512-d (InsightFace) or 128-d (SFace) float vectors leave.
"""

from __future__ import annotations
import logging
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

# ── optional imports ─────────────────────────────────────────────────────────
try:
    import insightface
    from insightface.app import FaceAnalysis
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

# ── YuNet / SFace model paths ─────────────────────────────────────────────────
_MODULE_DIR  = Path(__file__).parent
_MODELS_DIR  = _MODULE_DIR / "models"
_YUNET_PATH  = _MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_SFACE_PATH  = _MODELS_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/"
    "face_recognition_sface_2021dec.onnx"
)


class EmbeddingEngine:
    """
    Unified face detection + embedding extraction.

    Parameters
    ----------
    backend : "insightface" | "yunet_sface"
    gpu     : use GPU if available
    """

    BACKENDS = ("insightface", "yunet_sface")

    def __init__(self, backend: str = "insightface", gpu: bool = True):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}")
        self.backend = backend
        self.gpu     = gpu
        self._app    = None   # InsightFace FaceAnalysis
        self._yunet  = None   # cv2 YuNet
        self._sface  = None   # cv2 SFace
        self._ready  = False
        self._load()

    # ── public ───────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def extract(self, bgr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the largest face in bgr_frame and return its L2-normalised
        embedding vector.  Returns None if no face found or engine not ready.
        The input frame is never stored — only the embedding leaves this method.
        """
        if not self._ready or bgr_frame is None:
            return None
        try:
            if self.backend == "insightface":
                return self._extract_insightface(bgr_frame)
            return self._extract_yunet_sface(bgr_frame)
        except Exception as exc:
            log.warning("Embedding extraction error: %s", exc)
            return None

    def backend_label(self) -> str:
        if self.backend == "insightface":
            return "InsightFace buffalo_l  (ArcFace 512-d)"
        return "YuNet + SFace  (Apache 2.0, 128-d)"

    # ── loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if self.backend == "insightface":
            self._load_insightface()
        else:
            self._load_yunet_sface()

    def _load_insightface(self):
        if not _INSIGHTFACE_OK:
            log.warning(
                "insightface not installed (Python 3.13 needs VS C++ Build Tools to compile it).  "
                "Falling back to yunet_sface (Apache 2.0).  "
                "To install insightface: install VS C++ Build Tools first, then pip install insightface"
            )
            # Auto-fall-back to yunet_sface so the app still starts
            self.backend = 'yunet_sface'
            self._load_yunet_sface()
            return
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.gpu else ["CPUExecutionProvider"]
        )
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0 if self.gpu else -1, det_size=(640, 640))
        self._ready = True
        log.info("InsightFace buffalo_l loaded  (GPU=%s)", self.gpu)

    def _load_yunet_sface(self):
        if not _CV2_OK:
            log.error("opencv-contrib-python not installed.")
            return
        if not _YUNET_PATH.exists() or not _SFACE_PATH.exists():
            log.info("Downloading YuNet / SFace models…")
            self._auto_download()
        if _YUNET_PATH.exists() and _SFACE_PATH.exists():
            self._yunet = cv2.FaceDetectorYN.create(
                str(_YUNET_PATH), "", (320, 320),
                score_threshold=0.6, nms_threshold=0.3,
            )
            self._sface = cv2.FaceRecognizerSF.create(str(_SFACE_PATH), "")
            self._ready = True
            log.info("YuNet + SFace loaded")
        else:
            log.error("YuNet/SFace model files missing — download failed.")

    def _auto_download(self):
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for url, dst in [(YUNET_URL, _YUNET_PATH), (SFACE_URL, _SFACE_PATH)]:
            if not dst.exists():
                log.info("Downloading %s …", dst.name)
                try:
                    urllib.request.urlretrieve(url, dst)
                    log.info("  → saved to %s", dst)
                except Exception as exc:
                    log.error("  Download failed for %s: %s", dst.name, exc)

    # ── extraction ────────────────────────────────────────────────────────────

    def _extract_insightface(self, frame: np.ndarray) -> Optional[np.ndarray]:
        faces = self._app.get(frame)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding.astype(np.float32)   # already L2-normalised, (512,)

    def _extract_yunet_sface(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        face    = max(faces, key=lambda f: f[2] * f[3])   # largest by w*h
        aligned = self._sface.alignCrop(frame, face)
        emb     = self._sface.feature(aligned).flatten().astype(np.float32)
        norm    = np.linalg.norm(emb)
        if norm > 1e-8:
            emb /= norm
        return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fast cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))
