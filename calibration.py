# =============================================================================
# TAPIR Point Tracker v2 — Calibration & Kinematics
# PrepByEKS | Institut Sukan Negara Malaysia
#
# Two calibration types:
#   LinearCalibration  — user clicks 2 points, enters real-world distance (m)
#   FieldCalibration   — user clicks 4 corners, enters real-world W x H (m)
#                        Uses OpenCV perspective transform (homography)
#
# Kinematics pipeline (ISB-standard):
#   1. Convert pixel tracks  →  world-coordinate tracks (metres)
#   2. Savitzky-Golay deriv=1 × (1/dt)  →  velocity (m/s)
#   3. Savitzky-Golay deriv=2 × (1/dt²) →  acceleration (m/s²)
#   4. Speed = |velocity vector| (scalar)
#   5. Cumulative arc-length distance (m)
#
# Y-axis convention: origin at BOTTOM-LEFT of frame (screen Y is flipped so
# that upward motion → positive Y in CSV, matching biomechanics convention).
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("calibration")


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class LinearCalibration:
    """
    Scale factor derived from two pixel points and a known real-world distance.
    Coordinates stored as normalised [0,1] relative to display frame.
    """
    name:       str
    p0_norm:    tuple          # (x, y) normalised
    p1_norm:    tuple          # (x, y) normalised
    real_m:     float          # real-world distance in metres

    def px_per_m(self, frame_w: int, frame_h: int) -> float:
        """Pixels per metre for this calibration line."""
        dx = (self.p1_norm[0] - self.p0_norm[0]) * frame_w
        dy = (self.p1_norm[1] - self.p0_norm[1]) * frame_h
        px_dist = float(np.hypot(dx, dy))
        if px_dist < 1e-6 or self.real_m < 1e-6:
            return 1.0
        return px_dist / self.real_m

    def pixel_length_m(self, frame_w: int, frame_h: int) -> float:
        """Length of the calibration line in pixels (for display)."""
        dx = (self.p1_norm[0] - self.p0_norm[0]) * frame_w
        dy = (self.p1_norm[1] - self.p0_norm[1]) * frame_h
        return float(np.hypot(dx, dy))

    def to_world_xy(self, x_px: float, y_px: float,
                    frame_w: int, frame_h: int) -> tuple[float, float]:
        """
        Convert a single pixel point to world metres using the linear scale.
        Origin = p0 projected; X along calibration line, Y perpendicular.

        Y is flipped (frame_h - y_px) so that upward screen movement →
        positive world Y, matching biomechanics / bottom-left origin convention.
        """
        # Flip Y: screen top-left → bottom-left world origin
        y_px_flipped = frame_h - y_px

        ppm = self.px_per_m(frame_w, frame_h)
        if ppm < 1e-9:
            return (0.0, 0.0)
        # Normalised direction of calib line (negate dy for Y-flip)
        dx = (self.p1_norm[0] - self.p0_norm[0]) * frame_w
        dy = -((self.p1_norm[1] - self.p0_norm[1]) * frame_h)
        length = np.hypot(dx, dy)
        if length < 1e-6:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / length, dy / length
        # Perpendicular
        vx, vy = -uy, ux
        # Vector from p0 to target point (pixels, flipped Y)
        ox = self.p0_norm[0] * frame_w
        oy = frame_h - (self.p0_norm[1] * frame_h)
        rx, ry = x_px - ox, y_px_flipped - oy
        world_x = (rx * ux + ry * uy) / ppm
        world_y = (rx * vx + ry * vy) / ppm
        return (world_x, world_y)


@dataclass
class FieldCalibration:
    """
    Perspective (homography) calibration from a known planar quadrilateral.
    corners_norm: list of 4 (x,y) normalised points in order:
        [top-left, top-right, bottom-right, bottom-left]
    real_w, real_h: real-world dimensions in metres.
    """
    name:         str
    corners_norm: list          # 4 × (x, y) normalised
    real_w:       float         # metres
    real_h:       float         # metres
    _H:           Optional[np.ndarray] = field(default=None, repr=False)

    def _build_homography(self, frame_w: int, frame_h: int) -> np.ndarray:
        """
        Compute perspective transform pixel → world (metres).
        Input pixel Y is flipped (frame_h - y) so that bottom-left of frame
        maps to world origin and upward motion → positive Y.
        """
        src = np.array([
            [c[0] * frame_w, frame_h - c[1] * frame_h]
            for c in self.corners_norm
        ], dtype=np.float32)
        # After Y-flip TL→TR→BR→BL in screen space becomes BL→BR→TR→TL in world
        dst = np.array([
            [0.0,         self.real_h],
            [self.real_w, self.real_h],
            [self.real_w, 0.0        ],
            [0.0,         0.0        ],
        ], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        return H

    def to_world_xy(self, x_px: float, y_px: float,
                    frame_w: int, frame_h: int) -> tuple[float, float]:
        """
        Map a single pixel point to world metres.
        Y is flipped before transform so upward motion → positive world Y.
        """
        H = self._build_homography(frame_w, frame_h)
        if H is None:
            return (0.0, 0.0)
        y_px_flipped = frame_h - y_px
        pt = np.array([[[x_px, y_px_flipped]]], dtype=np.float32)
        wpt = cv2.perspectiveTransform(pt, H)
        return (float(wpt[0, 0, 0]), float(wpt[0, 0, 1]))

    def tracks_to_world(self, tracks_px: np.ndarray,
                        frame_w: int, frame_h: int) -> np.ndarray:
        """
        Vectorised conversion: (T, N, 2) pixel → (T, N, 2) world metres.
        Y is flipped before the homography so upward motion → positive world Y.
        """
        T, N, _ = tracks_px.shape
        H = self._build_homography(frame_w, frame_h)
        if H is None:
            return tracks_px.copy()
        flipped = tracks_px.copy().astype(np.float32)
        flipped[..., 1] = frame_h - flipped[..., 1]  # flip Y
        flat = flipped.reshape(-1, 1, 2)
        world_flat = cv2.perspectiveTransform(flat, H)
        return world_flat.reshape(T, N, 2).astype(np.float64)


# =============================================================================
# Calibration Manager
# =============================================================================

class CalibrationManager:
    """Stores all calibrations; assigns one per tracked point."""

    def __init__(self):
        self.linear: list[LinearCalibration] = []
        self.field:  list[FieldCalibration]  = []
        # per-point assignment: {point_index: ("linear"|"field", calib_index)}
        self.point_assignment: dict[int, tuple[str, int]] = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def add_linear(self, cal: LinearCalibration):
        self.linear.append(cal)

    def add_field(self, cal: FieldCalibration):
        self.field.append(cal)

    def remove_linear(self, idx: int):
        if 0 <= idx < len(self.linear):
            self.linear.pop(idx)
            # clean up assignments
            self.point_assignment = {
                k: v for k, v in self.point_assignment.items()
                if not (v[0] == "linear" and v[1] == idx)
            }

    def remove_field(self, idx: int):
        if 0 <= idx < len(self.field):
            self.field.pop(idx)
            self.point_assignment = {
                k: v for k, v in self.point_assignment.items()
                if not (v[0] == "field" and v[1] == idx)
            }

    def assign(self, point_idx: int, cal_type: str, cal_idx: int):
        self.point_assignment[point_idx] = (cal_type, cal_idx)

    def unassign(self, point_idx: int):
        self.point_assignment.pop(point_idx, None)

    def get_for_point(self, point_idx: int):
        """Return the calibration object assigned to this point, or None."""
        asgn = self.point_assignment.get(point_idx)
        if asgn is None:
            return None
        cal_type, cal_idx = asgn
        if cal_type == "linear" and cal_idx < len(self.linear):
            return self.linear[cal_idx]
        if cal_type == "field"  and cal_idx < len(self.field):
            return self.field[cal_idx]
        return None

    def all_names(self) -> list[str]:
        """Flat list: ['L: name', ..., 'F: name', ...]"""
        names = [f"L: {c.name}" for c in self.linear]
        names += [f"F: {c.name}" for c in self.field]
        return names

    def clear(self):
        self.linear.clear()
        self.field.clear()
        self.point_assignment.clear()

    def has_any(self) -> bool:
        return bool(self.linear or self.field)


# =============================================================================
# Kinematics
# =============================================================================

def _sg(signal: np.ndarray, window: int, poly: int,
        deriv: int = 0, delta: float = 1.0) -> np.ndarray:
    """
    Savitzky-Golay with safe parameter clamping.
    Falls back to finite differences if scipy unavailable.
    """
    T = len(signal)
    wl = min(window, T)
    if wl % 2 == 0:
        wl = max(3, wl - 1)
    po = min(poly, wl - 1)
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(signal, wl, po, deriv=deriv, delta=delta)
    except ImportError:
        if deriv == 0:
            return signal.copy()
        # Fallback: central finite differences
        out = np.gradient(signal, delta)
        if deriv == 2:
            out = np.gradient(out, delta)
        return out


def compute_kinematics(
    tracks_px:    np.ndarray,        # (T, N, 2) smoothed pixel coords
    occlusions:   np.ndarray,        # (T, N) bool
    calib_mgr:    CalibrationManager,
    frame_w:      int,
    frame_h:      int,
    fps:          float,
    sg_window:    int = 11,
    sg_poly:      int = 3,
) -> dict:
    """
    Compute kinematics for every tracked point using its assigned calibration.

    Y-axis is flipped inside to_world_xy / tracks_to_world so that upward
    screen movement produces positive world Y values in the output arrays.

    Returns dict:
        world_x      (T, N)  metres
        world_y      (T, N)  metres
        vel_x        (T, N)  m/s
        vel_y        (T, N)  m/s
        speed        (T, N)  m/s  (|velocity|)
        accel        (T, N)  m/s² (d(speed)/dt)
        distance     (T, N)  metres (cumulative arc-length)
        calib_names  list[str | None]  — calibration name per point
    """
    T, N, _ = tracks_px.shape
    dt = 1.0 / max(fps, 1e-6)

    world_x    = np.zeros((T, N))
    world_y    = np.zeros((T, N))
    vel_x      = np.zeros((T, N))
    vel_y      = np.zeros((T, N))
    speed      = np.zeros((T, N))
    accel      = np.zeros((T, N))
    distance   = np.zeros((T, N))
    calib_names = []

    for n in range(N):
        cal = calib_mgr.get_for_point(n)
        calib_names.append(cal.name if cal is not None else None)

        if cal is None:
            # No calibration — skip (leave zeros)
            continue

        # ── 1. Convert to world coords ────────────────────────────────────
        if isinstance(cal, FieldCalibration):
            # Vectorised homography (Y-flip handled inside tracks_to_world)
            pt_px = tracks_px[:, n, :]               # (T, 2)
            flat  = pt_px.reshape(-1, 1, 2).astype(np.float32)
            flat_flipped = flat.copy()
            flat_flipped[:, 0, 1] = frame_h - flat_flipped[:, 0, 1]  # flip Y
            H = cal._build_homography(frame_w, frame_h)
            w_flat = cv2.perspectiveTransform(flat_flipped, H)
            wx = w_flat[:, 0, 0].astype(np.float64)
            wy = w_flat[:, 0, 1].astype(np.float64)
        else:  # LinearCalibration — Y-flip handled in to_world_xy
            ppm = cal.px_per_m(frame_w, frame_h)
            ox  = cal.p0_norm[0] * frame_w
            oy  = frame_h - (cal.p0_norm[1] * frame_h)  # flipped origin
            dx_line = (cal.p1_norm[0] - cal.p0_norm[0]) * frame_w
            dy_line = -((cal.p1_norm[1] - cal.p0_norm[1]) * frame_h)  # flip
            length  = np.hypot(dx_line, dy_line)
            if length < 1e-6:
                ux, uy, vx_dir, vy_dir = 1.0, 0.0, 0.0, 1.0
            else:
                ux, uy = dx_line / length, dy_line / length
                vx_dir, vy_dir = -uy, ux
            rx = tracks_px[:, n, 0] - ox
            ry = (frame_h - tracks_px[:, n, 1]) - oy  # flip Y before offset
            wx = (rx * ux + ry * uy) / ppm
            wy = (rx * vx_dir + ry * vy_dir) / ppm

        # Mask occluded frames with interpolated values
        occ = occlusions[:, n].astype(bool)
        for coord in [wx, wy]:
            visible = np.where(~occ)[0]
            if len(visible) >= 2:
                occ_idx = np.where(occ)[0]
                coord[occ_idx] = np.interp(occ_idx, visible, coord[visible])

        world_x[:, n] = wx
        world_y[:, n] = wy

        # ── 2. Velocity (ISB: SG deriv=1) ────────────────────────────────
        vx_n = _sg(wx, sg_window, sg_poly, deriv=1, delta=dt)
        vy_n = _sg(wy, sg_window, sg_poly, deriv=1, delta=dt)
        vel_x[:, n] = vx_n
        vel_y[:, n] = vy_n

        # ── 3. Speed ─────────────────────────────────────────────────────
        spd = np.hypot(vx_n, vy_n)
        speed[:, n] = spd

        # ── 4. Acceleration (SG deriv=1 of speed) ────────────────────────
        accel[:, n] = _sg(spd, sg_window, sg_poly, deriv=1, delta=dt)

        # ── 5. Cumulative arc-length distance ─────────────────────────────
        dx = np.diff(wx, prepend=wx[0])
        dy = np.diff(wy, prepend=wy[0])
        step = np.hypot(dx, dy)
        step[occ] = 0.0          # don't accumulate during occlusion
        distance[:, n] = np.cumsum(step)

    return {
        "world_x":    world_x,
        "world_y":    world_y,
        "vel_x":      vel_x,
        "vel_y":      vel_y,
        "speed":      speed,
        "accel":      accel,
        "distance":   distance,
        "calib_names": calib_names,
    }
