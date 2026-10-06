# =============================================================================
# TAPIR Point Tracker v2 — Visualizer
# PrepByEKS | Institut Sukan Negara Malaysia
# =============================================================================

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass
from config import COLORS, LINE_COLORS, POINT_RADIUS, TRAIL_LENGTH, TRAIL_ALPHA


# -----------------------------------------------------------------------------
# Line connection descriptor
# -----------------------------------------------------------------------------

@dataclass
class LineConnection:
    idx_a:     int
    idx_b:     int
    color_bgr: tuple = (255, 255, 255)
    thickness: int   = 2
    label:     str   = ""
    dashed:    bool  = False


# -----------------------------------------------------------------------------
# Point trails & markers
# -----------------------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    frame_idx: int,
    tracks: np.ndarray,           # (T, N, 2) — smoothed/raw x,y pixels
    occlusions: np.ndarray,       # (T, N) bool
    scores: np.ndarray,           # (T, N) float
    labels: list = None,
    connections: list = None,
    show_trails: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    out     = frame.copy()
    T, N, _ = tracks.shape

    # -- Line connections (drawn before points so points sit on top) ----------
    if connections:
        for conn in connections:
            a, b = conn.idx_a, conn.idx_b
            if a >= N or b >= N or frame_idx >= T:
                continue
            if occlusions[frame_idx, a] or occlusions[frame_idx, b]:
                continue
            xa, ya = int(tracks[frame_idx, a, 0]), int(tracks[frame_idx, a, 1])
            xb, yb = int(tracks[frame_idx, b, 0]), int(tracks[frame_idx, b, 1])
            if conn.dashed:
                _draw_dashed_line(out, (xa, ya), (xb, yb), conn.color_bgr, conn.thickness)
            else:
                cv2.line(out, (xa, ya), (xb, yb), conn.color_bgr, conn.thickness, cv2.LINE_AA)
            if conn.label and show_labels:
                mx, my = (xa + xb) // 2, (ya + yb) // 2
                cv2.putText(out, conn.label, (mx + 3, my - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, conn.color_bgr, 1, cv2.LINE_AA)

    # -- Trails & points -------------------------------------------------------
    for n in range(N):
        color = COLORS[n % len(COLORS)]

        if show_trails:
            trail_start = max(0, frame_idx - TRAIL_LENGTH)
            trail_pts   = []
            for t in range(trail_start, min(frame_idx + 1, T)):
                if not occlusions[t, n]:
                    trail_pts.append((int(tracks[t, n, 0]), int(tracks[t, n, 1])))
            if len(trail_pts) >= 2:
                overlay = out.copy()
                for i in range(1, len(trail_pts)):
                    alpha = TRAIL_ALPHA * (i / len(trail_pts))
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(overlay, trail_pts[i - 1], trail_pts[i], faded, 2, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        if frame_idx >= T:
            continue

        if not occlusions[frame_idx, n]:
            cx, cy = int(tracks[frame_idx, n, 0]), int(tracks[frame_idx, n, 1])
            score  = float(scores[frame_idx, n])
            r = max(1, POINT_RADIUS)  # 1/3 of original via config (was 5 -> now 2)
            # Semi-transparent filled dot (50% alpha)
            dot_layer = out.copy()
            cv2.circle(dot_layer, (cx, cy), r,     color, -1, cv2.LINE_AA)
            cv2.circle(dot_layer, (cx, cy), r + 2, color,  1, cv2.LINE_AA)
            cv2.addWeighted(dot_layer, 0.5, out, 0.5, 0, out)
            if show_labels:
                tag = labels[n] if labels and n < len(labels) else f"P{n+1}"
                cv2.putText(out, f"{tag} {score:.2f}",
                            (cx + r + 3, cy - r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            # Hollow ring at last known position when occluded
            if frame_idx > 0:
                prev = frame_idx - 1
                while prev >= 0 and occlusions[prev, n]:
                    prev -= 1
                if prev >= 0:
                    cx, cy = int(tracks[prev, n, 0]), int(tracks[prev, n, 1])
                    cv2.circle(out, (cx, cy), max(1, POINT_RADIUS) + 1,
                               tuple(c // 2 for c in color), 1, cv2.LINE_AA)

    return out


def _draw_dashed_line(img, p0, p1, color, thickness, dash=8, gap=5):
    x0, y0 = p0; x1, y1 = p1
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if dist < 1:
        return
    dx, dy = (x1 - x0) / dist, (y1 - y0) / dist
    pos = 0.0
    while pos < dist:
        start = (int(x0 + dx * pos),              int(y0 + dy * pos))
        end   = (int(x0 + dx * min(pos + dash, dist)), int(y0 + dy * min(pos + dash, dist)))
        cv2.line(img, start, end, color, thickness, cv2.LINE_AA)
        pos += dash + gap


# -----------------------------------------------------------------------------
# HUD overlay
# -----------------------------------------------------------------------------

def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    n_points: int,
    status: str = "",
    smoothed: bool = False,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    bar = out.copy()
    cv2.rectangle(bar, (0, 0), (w, 28), (10, 10, 10), -1)
    cv2.addWeighted(bar, 0.65, out, 0.35, 0, out)

    smooth_tag = "  [smoothed]" if smoothed else ""
    timecode   = f"{frame_idx / max(fps, 1):.2f}s / {total_frames / max(fps, 1):.2f}s"
    label      = f"PrepByEKS | TAPIR  [{timecode}]  Pts:{n_points}{smooth_tag}  {status}"
    cv2.putText(out, label, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 150), 1, cv2.LINE_AA)

    pct = int((frame_idx / max(total_frames - 1, 1)) * w)
    cv2.rectangle(out, (0, h - 3), (pct, h), (0, 200, 150), -1)

    return out


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def frames_to_preview(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w  = frame.shape[:2]
    # Guard: canvas may report tiny dimensions before layout completes
    max_w = max(max_w, 1)
    max_h = max(max_h, 1)
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        tw = max(1, int(w * scale))
        th = max(1, int(h * scale))
        frame = cv2.resize(frame, (tw, th),
                           interpolation=cv2.INTER_AREA)
    return frame
