# =============================================================================
# TAPIR Point Tracker v2 — Smoothing & Interpolation
# PrepByEKS | Institut Sukan Negara Malaysia
#
# Pipeline:
#   1. interpolate_occlusions  -> fill gaps where model lost the point
#   2. smooth_tracks           -> Savitzky-Golay per coordinate axis
# =============================================================================

from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger("smoothing")


# -----------------------------------------------------------------------------
# Step 1 - Occlusion Interpolation
# -----------------------------------------------------------------------------

def interpolate_occlusions(
    tracks: np.ndarray,       # (T, N, 2)  x,y pixels
    occlusions: np.ndarray,   # (T, N)     bool - True = occluded
) -> np.ndarray:
    """
    Linearly interpolate x,y for frames where a point is occluded.
    Uses the last visible position before and first visible position after
    the gap to fill in plausible in-between coordinates.
    Extrapolates at start/end if no anchor exists on one side.
    Returns a new (T, N, 2) array.
    """
    T, N, _ = tracks.shape
    out = tracks.copy()

    for n in range(N):
        occ_flags = occlusions[:, n].astype(bool)
        if not occ_flags.any():
            continue

        visible_frames = np.where(~occ_flags)[0]
        if len(visible_frames) == 0:
            continue

        for t in np.where(occ_flags)[0]:
            before = visible_frames[visible_frames < t]
            after  = visible_frames[visible_frames > t]

            if len(before) > 0 and len(after) > 0:
                t0, t1 = before[-1], after[0]
                alpha  = (t - t0) / (t1 - t0)
                out[t, n] = (1 - alpha) * tracks[t0, n] + alpha * tracks[t1, n]
            elif len(before) > 0:
                out[t, n] = tracks[before[-1], n]
            else:
                out[t, n] = tracks[after[0], n]

    return out


# -----------------------------------------------------------------------------
# Step 2 - Savitzky-Golay Smoothing
# -----------------------------------------------------------------------------

def smooth_tracks(
    tracks: np.ndarray,      # (T, N, 2)
    window_length: int = 11,
    polyorder: int    = 3,
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter along the time axis for each point and axis.
    Falls back to a simple moving average if scipy is not available.
    window_length is clamped to an odd value <= T.
    """
    T, N, _ = tracks.shape
    out = tracks.copy()

    wl = min(window_length, T)
    if wl % 2 == 0:
        wl = max(1, wl - 1)
    po = min(polyorder, wl - 1)

    try:
        from scipy.signal import savgol_filter  # type: ignore
        for n in range(N):
            for ax in range(2):
                out[:, n, ax] = savgol_filter(tracks[:, n, ax], wl, po)
        logger.debug("Smoothing: Savitzky-Golay (window=%d, poly=%d)", wl, po)
    except ImportError:
        logger.warning("scipy not available -- falling back to moving average smoothing.")
        out = _moving_average(tracks, wl)

    return out


def _moving_average(tracks: np.ndarray, window: int) -> np.ndarray:
    """Simple symmetric moving average fallback."""
    T, N, _ = tracks.shape
    out  = tracks.copy()
    half = window // 2
    for n in range(N):
        for ax in range(2):
            signal = tracks[:, n, ax]
            for t in range(T):
                lo = max(0, t - half)
                hi = min(T, t + half + 1)
                out[t, n, ax] = signal[lo:hi].mean()
    return out


# -----------------------------------------------------------------------------
# Combined pipeline
# -----------------------------------------------------------------------------

def process_tracks(
    tracks: np.ndarray,
    occlusions: np.ndarray,
    interp: bool = True,
    smooth: bool = True,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Full post-processing pipeline.
    Returns smoothed (T, N, 2) track array.
    Occlusions array is not modified.
    """
    result = tracks.copy()
    if interp:
        result = interpolate_occlusions(result, occlusions)
    if smooth:
        result = smooth_tracks(result, window_length, polyorder)
    return result
