# =============================================================================
# TAPIR Point Tracker v2 — Annotation Manager
# PrepByEKS | Institut Sukan Negara Malaysia
#
# Stores per-frame annotations (lines, arrows, rects, ellipses, text, angles).
# All coordinates stored normalised [0,1] relative to the original frame dims.
# Rendered at display time by draw_annotations().
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class AnnotLine:
    x0: float; y0: float; x1: float; y1: float
    color: tuple = (0, 200, 150); width: int = 2

@dataclass
class AnnotArrow:
    x0: float; y0: float; x1: float; y1: float
    color: tuple = (0, 200, 150); width: int = 2

@dataclass
class AnnotRect:
    x0: float; y0: float; x1: float; y1: float
    color: tuple = (0, 200, 150); width: int = 2; fill: bool = False

@dataclass
class AnnotEllipse:
    cx: float; cy: float; rx: float; ry: float
    color: tuple = (0, 200, 150); width: int = 2; fill: bool = False

@dataclass
class AnnotText:
    x: float; y: float; text: str
    color: tuple = (0, 200, 150); size: float = 0.55; bold: bool = False

@dataclass
class AnnotAngle:
    # Three points: vertex at p1, rays toward p0 and p2
    x0: float; y0: float
    x1: float; y1: float   # vertex
    x2: float; y2: float
    color: tuple = (0, 200, 150); width: int = 2

    def degrees(self) -> float:
        v1 = np.array([self.x0 - self.x1, self.y0 - self.y1])
        v2 = np.array([self.x2 - self.x1, self.y2 - self.y1])
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


# -----------------------------------------------------------------------------
# Manager
# -----------------------------------------------------------------------------

class AnnotationManager:
    """Per-frame annotation store."""

    def __init__(self):
        self._data: dict[int, list] = {}
        self._global: list = []    # shown on ALL frames

    def add(self, frame_idx: int, obj, global_annot: bool = False):
        if global_annot:
            self._global.append(obj)
        else:
            self._data.setdefault(frame_idx, []).append(obj)

    def get(self, frame_idx: int) -> list:
        return self._global + self._data.get(frame_idx, [])

    def pop_last(self, frame_idx: int) -> bool:
        """Undo last annotation on this frame (or global)."""
        lst = self._data.get(frame_idx, [])
        if lst:
            lst.pop()
            return True
        if self._global:
            self._global.pop()
            return True
        return False

    def clear_frame(self, frame_idx: int):
        self._data.pop(frame_idx, None)

    def clear_all(self):
        self._data.clear()
        self._global.clear()

    def has_any(self) -> bool:
        return bool(self._data or self._global)


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------

def hex_to_bgr(hex_str: str) -> tuple:
    hex_str = hex_str.lstrip("#")
    r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    return (b, g, r)


def draw_annotations(frame: np.ndarray, annotations: list) -> np.ndarray:
    """Render annotation objects onto a BGR frame copy."""
    if not annotations:
        return frame
    H, W = frame.shape[:2]
    out  = frame.copy()

    def px(xn, yn):
        return int(xn * W), int(yn * H)

    for a in annotations:
        if isinstance(a, AnnotLine):
            cv2.line(out, px(a.x0, a.y0), px(a.x1, a.y1),
                     a.color, a.width, cv2.LINE_AA)

        elif isinstance(a, AnnotArrow):
            p0, p1 = px(a.x0, a.y0), px(a.x1, a.y1)
            cv2.arrowedLine(out, p0, p1, a.color, a.width,
                            cv2.LINE_AA, tipLength=0.08)

        elif isinstance(a, AnnotRect):
            p0, p1 = px(a.x0, a.y0), px(a.x1, a.y1)
            thick  = -1 if a.fill else a.width
            cv2.rectangle(out, p0, p1, a.color, thick, cv2.LINE_AA)

        elif isinstance(a, AnnotEllipse):
            c    = px(a.cx, a.cy)
            axes = (max(1, int(a.rx * W)), max(1, int(a.ry * H)))
            thick = -1 if a.fill else a.width
            cv2.ellipse(out, c, axes, 0, 0, 360, a.color, thick, cv2.LINE_AA)

        elif isinstance(a, AnnotText):
            face  = cv2.FONT_HERSHEY_SIMPLEX
            thick = 2 if a.bold else 1
            pt    = px(a.x, a.y)
            cv2.putText(out, a.text, pt, face, a.size, (0, 0, 0), thick+1, cv2.LINE_AA)
            cv2.putText(out, a.text, pt, face, a.size, a.color,   thick,   cv2.LINE_AA)

        elif isinstance(a, AnnotAngle):
            p0 = px(a.x0, a.y0)
            p1 = px(a.x1, a.y1)   # vertex
            p2 = px(a.x2, a.y2)
            cv2.line(out, p1, p0, a.color, a.width, cv2.LINE_AA)
            cv2.line(out, p1, p2, a.color, a.width, cv2.LINE_AA)
            deg = a.degrees()
            cv2.putText(out, f"{deg:.1f}°",
                        (p1[0]+6, p1[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, a.color, 1, cv2.LINE_AA)

    return out
