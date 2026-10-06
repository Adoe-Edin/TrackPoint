# =============================================================================
# TAPIR Point Tracker v2 — Exporter
# PrepByEKS | Institut Sukan Negara Malaysia
# =============================================================================

import csv
import os
import numpy as np
from datetime import datetime
from config import OUTPUT_DIR, CSV_DELIMITER, EXPORT_CSV


def export_tracks_csv(
    tracks: np.ndarray,
    occlusions: np.ndarray,
    scores: np.ndarray,
    fps: float,
    video_name: str,
    labels: list = None,
) -> str:
    """
    Write tracking data to CSV.
    Columns: frame, time_s, point_id, label, x_px, y_px, visible, score
    """
    if not EXPORT_CSV:
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    base     = os.path.splitext(os.path.basename(video_name))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_tracks_{stamp}.csv")

    T, N, _ = tracks.shape

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=CSV_DELIMITER)
        writer.writerow(["# PrepByEKS — TAPIR Point Tracker v2"])
        writer.writerow(["# Source", video_name])
        writer.writerow(["# Exported", datetime.now().isoformat()])
        writer.writerow(["# Points", N])
        writer.writerow([])
        writer.writerow(["frame", "time_s", "point_id", "label",
                         "x_px", "y_px", "visible", "score"])

        for t in range(T):
            for n in range(N):
                label   = labels[n] if labels and n < len(labels) else f"P{n+1}"
                x, y    = tracks[t, n]
                visible = int(not occlusions[t, n])
                score   = float(scores[t, n])
                writer.writerow([
                    t, f"{t / fps:.4f}", n + 1, label,
                    f"{x:.2f}", f"{y:.2f}", visible, f"{score:.4f}",
                ])

    return out_path


def export_kinematics_csv(
    kinematics: dict,
    fps: float,
    video_name: str,
    labels: list = None,
) -> str:
    """
    Write kinematics data to CSV.

    Columns:
        frame, time_s, point_id, label, calib,
        world_x_m, world_y_m,
        vel_x_ms, vel_y_ms, speed_ms,
        accel_ms2, distance_m
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    base     = os.path.splitext(os.path.basename(video_name))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_kinematics_{stamp}.csv")

    world_x     = kinematics["world_x"]      # (T, N)
    world_y     = kinematics["world_y"]
    vel_x       = kinematics["vel_x"]
    vel_y       = kinematics["vel_y"]
    speed       = kinematics["speed"]
    accel       = kinematics["accel"]
    distance    = kinematics["distance"]
    calib_names = kinematics["calib_names"]  # list[str|None] length N

    T, N = world_x.shape

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=CSV_DELIMITER)
        writer.writerow(["# PrepByEKS — TAPIR Point Tracker v2 — Kinematics"])
        writer.writerow(["# Source", video_name])
        writer.writerow(["# Exported", datetime.now().isoformat()])
        writer.writerow(["# FPS", f"{fps:.4f}"])
        writer.writerow(["# Points", N])
        writer.writerow([])
        writer.writerow([
            "frame", "time_s", "point_id", "label", "calibration",
            "world_x_m", "world_y_m",
            "vel_x_ms", "vel_y_ms", "speed_ms",
            "accel_ms2", "distance_m",
        ])

        for t in range(T):
            for n in range(N):
                label      = labels[n] if labels and n < len(labels) else f"P{n+1}"
                calib_name = calib_names[n] if n < len(calib_names) else ""
                if calib_name is None:
                    calib_name = "none"
                writer.writerow([
                    t,
                    f"{t / fps:.4f}",
                    n + 1,
                    label,
                    calib_name,
                    f"{world_x[t, n]:.4f}",
                    f"{world_y[t, n]:.4f}",
                    f"{vel_x[t, n]:.4f}",
                    f"{vel_y[t, n]:.4f}",
                    f"{speed[t, n]:.4f}",
                    f"{accel[t, n]:.4f}",
                    f"{distance[t, n]:.4f}",
                ])

    return out_path
