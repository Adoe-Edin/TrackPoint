# =============================================================================
# TAPIR Point Tracker v2 — Video I/O
# PrepByEKS | Institut Sukan Negara Malaysia
# =============================================================================

import cv2
import numpy as np
import os
import gc
from config import OUTPUT_DIR, VIDEO_CODEC, EXPORT_VIDEO, DISPLAY_MAX_W, DISPLAY_MAX_H


def _safe_resize(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Downscale frame to fit within max_w x max_h, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return frame


class VideoReader:
    """
    Wrapper around cv2.VideoCapture.
    Frames are downscaled at read-time to DISPLAY_MAX_W x DISPLAY_MAX_H to
    reduce RAM consumption (~4x less memory than 1080p originals).
    Original resolution is preserved in self.orig_width / self.orig_height
    for correct coordinate mapping.
    """

    def __init__(self, path: str):
        self.path = path
        self.cap  = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
        self.orig_width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Compute display resolution (used for frame storage and canvas display)
        scale = min(DISPLAY_MAX_W / self.orig_width,
                    DISPLAY_MAX_H / self.orig_height, 1.0)
        self.width  = int(self.orig_width  * scale)
        self.height = int(self.orig_height * scale)
        self._scale = scale
        self._cache: dict = {}

    def read_frame(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self._scale < 1.0:
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)
        self._cache[idx] = frame
        return frame

    def read_all_frames(self, callback=None) -> list:
        """
        Read every frame, downscaling immediately to save memory.
        Runs GC every 100 frames to prevent fragmentation.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        i = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            try:
                if self._scale < 1.0:
                    frame = cv2.resize(frame, (self.width, self.height),
                                       interpolation=cv2.INTER_AREA)
                frames.append(frame)
            except cv2.error as e:
                raise MemoryError(
                    f"OpenCV out of memory at frame {i}.\n"
                    f"Try reducing DISPLAY_MAX_W / DISPLAY_MAX_H in config.py.\n"
                    f"Original error: {e}"
                )
            if callback:
                callback(i, self.frame_count)
            i += 1
            if i % 100 == 0:
                gc.collect()
        return frames

    def get_preview_frame(self, idx: int = 0):
        return self.read_frame(idx)

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def close(self):
        self.cap.release()
        self._cache.clear()
        gc.collect()


class VideoWriter:
    """Write annotated output video at display resolution."""

    def __init__(self, source: VideoReader, suffix: str = "_tapir_tracked"):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base     = os.path.splitext(os.path.basename(source.path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}{suffix}.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        self.writer = cv2.VideoWriter(
            out_path, fourcc, source.fps,
            (source.width, source.height)
        )
        self.out_path = out_path

    def write(self, frame: np.ndarray):
        if EXPORT_VIDEO:
            self.writer.write(frame)

    def close(self):
        self.writer.release()


def resize_for_model(frame: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def frames_to_rgb_tensor(frames: list) -> np.ndarray:
    """
    Convert list of BGR frames -> (T, H, W, 3) float32 RGB array [0,1].
    Processes in chunks to avoid a single massive allocation.
    """
    chunk_size = 50
    parts = []
    for i in range(0, len(frames), chunk_size):
        chunk = frames[i:i + chunk_size]
        rgb   = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in chunk]
        parts.append(np.stack(rgb, axis=0).astype(np.float32) / 255.0)
        gc.collect()
    return np.concatenate(parts, axis=0)
