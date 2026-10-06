# =============================================================================
# TAPIR Point Tracker v2 -- Inference Engine
# PrepByEKS | Institut Sukan Negara Malaysia
# =============================================================================

from __future__ import annotations
import numpy as np
import torch
import logging
from dataclasses import dataclass
from config import (
    DEVICE, MIXED_PRECISION, TAPIR_CHECKPOINT, MODEL_BACKEND,
    RESIZE_HEIGHT, RESIZE_WIDTH, CHUNK_SIZE, OCCLUSION_THRESHOLD,
    COTRACKER_WINDOW_SIZE, TAPIR_NUM_PIPS_ITER,
    COTRACKER_CHUNK_FRAMES, COTRACKER_CHUNK_OVERLAP,
    COTRACKER_INFER_H, COTRACKER_INFER_W,
)

logger = logging.getLogger("tapir_engine")


@dataclass
class TrackResult:
    """Output from one tracking run."""
    tracks:     np.ndarray   # (T, N, 2)  x,y in original frame coords
    occlusions: np.ndarray   # (T, N)     True = occluded
    scores:     np.ndarray   # (T, N)     visibility confidence [0,1]


def _import_tapir_model():
    """
    Import tapnet.torch.tapir_model while bypassing the JAX-heavy
    tapnet/__init__.py by pre-stubbing all modules it tries to import.
    """
    import sys
    import types

    _jax_stubs = [
        "einops",
        "tapnet.models",
        "tapnet.models.tapir_model",
        "tapnet.models.tapnet_model",
        "tapnet.robotap",
        "tapnet.robotap.tapir_clustering",
        "tapnet.tapvid",
        "tapnet.tapvid.evaluation_datasets",
    ]
    _inserted = []
    for _mod in _jax_stubs:
        if _mod not in sys.modules:
            sys.modules[_mod] = types.ModuleType(_mod)
            _inserted.append(_mod)

    try:
        from tapnet.torch import tapir_model  # type: ignore
        return tapir_model
    finally:
        for _mod in _inserted:
            sys.modules.pop(_mod, None)


# =============================================================================
# TAPIR Engine (primary)
# =============================================================================

class TAPIREngine:

    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.model  = None
        self._load()

    def release(self):
        """Move model off GPU and free CUDA memory."""
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TAPIREngine released.")

    def __del__(self):
        self.release()

    def _load(self):
        try:
            tapir_model = _import_tapir_model()
            # pyramid_level=0, extra_convs=False matches tapir_checkpoint_panning.pt
            self.model = tapir_model.TAPIR(pyramid_level=0, extra_convs=False,
                                           num_pips_iter=TAPIR_NUM_PIPS_ITER)
            import os
            if TAPIR_CHECKPOINT and os.path.exists(TAPIR_CHECKPOINT):
                ckpt = torch.load(TAPIR_CHECKPOINT, map_location=self.device,
                                  weights_only=False)
                # tapir_checkpoint_panning.pt is a raw OrderedDict state dict
                state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state)
                logger.info("TAPIR checkpoint loaded: %s", TAPIR_CHECKPOINT)
            else:
                logger.warning(
                    "No checkpoint found -- running with random weights. "
                    "Download: https://storage.googleapis.com/dm-tapnet/"
                    "tapir_checkpoint_panning.pt"
                )
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError(
                "tapnet not installed. Run:\n"
                "  pip install git+https://github.com/google-deepmind/tapnet.git"
            )

    @torch.no_grad()
    def track(self, frames_rgb: np.ndarray, query_points: np.ndarray,
              orig_hw: tuple) -> TrackResult:
        import cv2
        frames_resized = np.stack([
            cv2.resize(f, (RESIZE_WIDTH, RESIZE_HEIGHT)) for f in frames_rgb
        ])
        # TAPIR expects (B, T, H, W, C) float32 [0,1]
        video_t = (torch.from_numpy(frames_resized)
                   .float().to(self.device).unsqueeze(0))

        # query_points: (N, 3) with [t_abs, y_px, x_px] in resized-frame space
        qp = query_points.copy()
        qp[:, 1] *= RESIZE_HEIGHT   # yn -> y pixel in resized frame
        qp[:, 2] *= RESIZE_WIDTH    # xn -> x pixel in resized frame
        # qp[:, 0] is already the absolute frame index (float)
        qp_t = torch.from_numpy(qp).float().to(self.device).unsqueeze(0)

        with torch.autocast(device_type="cuda",
                            enabled=MIXED_PRECISION and DEVICE == "cuda"):
            # refinement_resolutions=None: TAPIR auto-selects based on input size
            # Avoids the extra VRAM cost of a forced full-res refinement pass
            outputs = self.model(video_t, qp_t)

        # TAPIR returns tracks (B, N, T, 2) -- transpose to (T, N, 2) for draw_tracks
        tracks_raw = outputs["tracks"][0]                        # (N, T, 2)
        tracks     = tracks_raw.permute(1, 0, 2).cpu().numpy()  # -> (T, N, 2)

        # occlusion logit: higher = more likely OCCLUDED (per TAPIR docstring)
        # visible_prob = sigmoid(-logit): high value means VISIBLE
        occ_logit  = outputs["occlusion"][0].cpu().numpy()  # (N, T)
        visible    = 1.0 / (1.0 + np.exp(occ_logit))        # (N, T), high = visible
        scores     = visible.T                                # -> (T, N)
        occlusions = scores < OCCLUSION_THRESHOLD            # True when not visible enough

        # Scale tracks back to original frame coords
        oH, oW = orig_hw
        tracks[..., 0] *= oW / RESIZE_WIDTH
        tracks[..., 1] *= oH / RESIZE_HEIGHT

        return TrackResult(tracks=tracks, occlusions=occlusions, scores=scores)


# =============================================================================
# CoTracker Engine  (Meta)
# =============================================================================

class CoTrackerEngine:

    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        self.model  = None
        self._load()

    def release(self):
        """Move model off GPU and free CUDA memory."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CoTrackerEngine released.")

    def __del__(self):
        self.release()

    def _load(self):
        # Redirect torch.hub cache to checkpoints\torch_cache so install.bat
        # pre-populated weights are used and no download occurs at first launch.
        import os as _os, pathlib as _pl
        _cache_dir = _pl.Path(__file__).parent / "checkpoints" / "torch_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _os.environ.setdefault("TORCH_HOME", str(_cache_dir))

        model  = None
        errors = []

        try:
            model = torch.hub.load(
                "facebookresearch/co-tracker",
                "cotracker3_offline",
                trust_repo=True,
                window_len=COTRACKER_WINDOW_SIZE,
            )
            logger.info("CoTracker loaded via torch.hub (pretrained weights).")
        except Exception as e:
            errors.append(f"torch.hub: {e}")

        if model is None:
            try:
                from cotracker.predictor import CoTrackerPredictor   # type: ignore
                import os, urllib.request
                ckpt_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
                ckpt_path = os.path.join(ckpt_dir, "cotracker3_offline.pth")
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_dir, exist_ok=True)
                    url = ("https://huggingface.co/facebook/cotracker3/resolve/main/"
                           "scaled_offline.pth")
                    logger.info("Downloading CoTracker checkpoint to %s ...", ckpt_path)
                    urllib.request.urlretrieve(url, ckpt_path)
                model = CoTrackerPredictor(checkpoint=ckpt_path, offline=True)
                logger.info("CoTracker loaded via CoTrackerPredictor (checkpoint: %s).", ckpt_path)
            except Exception as e:
                errors.append(f"CoTrackerPredictor+checkpoint: {e}")

        if model is None:
            try:
                import cotracker                                  # type: ignore
                model = cotracker.cotracker3_offline()
                logger.info("CoTracker loaded via legacy API.")
            except Exception as e:
                errors.append(f"legacy API: {e}")

        if model is None:
            raise ImportError(
                "CoTracker could not be loaded. Tried:\n" +
                "\n".join(f"  - {e}" for e in errors) +
                "\n\nInstall with:\n"
                "  pip install git+https://github.com/facebookresearch/co-tracker.git"
            )

        self.model = model
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def track(self, frames_rgb: np.ndarray, query_points: np.ndarray,
              orig_hw: tuple) -> TrackResult:
        oH, oW = orig_hw
        T      = frames_rgb.shape[0]
        N      = query_points.shape[0]

        qp = query_points.copy()
        queries_px = np.stack(
            [qp[:, 0], qp[:, 2] * oW, qp[:, 1] * oH], axis=-1
        )  # (N, 3) -- [frame_idx, x_px, y_px]

        # -- Pre-resize to inference resolution (~7x faster) ------------------
        iH, iW   = COTRACKER_INFER_H, COTRACKER_INFER_W
        scale_x  = oW / iW
        scale_y  = oH / iH
        import cv2 as _cv2
        frames_infer = np.stack([
            _cv2.resize(f, (iW, iH), interpolation=_cv2.INTER_AREA)
            for f in frames_rgb
        ]).astype(np.float32)  # (T, iH, iW, 3)
        q_infer        = queries_px.copy()
        q_infer[:, 1] /= scale_x   # x_px -> x_infer
        q_infer[:, 2] /= scale_y   # y_px -> y_infer

        # -- Chunk processing: prevents OOM / freeze on long videos -----------
        chunk_sz = COTRACKER_CHUNK_FRAMES
        overlap  = COTRACKER_CHUNK_OVERLAP

        if T <= chunk_sz:
            # Short video: process all at once
            video_t = (torch.from_numpy(frames_infer)
                       .permute(0, 3, 1, 2).unsqueeze(0)
                       .float().to(self.device))
            queries_t = (torch.from_numpy(q_infer)
                         .float().unsqueeze(0).to(self.device))
            pred_tracks, pred_vis = self.model(video_t, queries=queries_t)
            all_tracks = pred_tracks[0].cpu().numpy()              # (T, N, 2)
            all_vis    = torch.sigmoid(pred_vis[0]).cpu().numpy()  # (T, N)
        else:
            # Long video: process in overlapping chunks and stitch
            logger.info("CoTracker: chunking %d frames (chunk=%d overlap=%d)",
                        T, chunk_sz, overlap)
            all_tracks = np.zeros((T, N, 2), dtype=np.float32)
            all_vis    = np.zeros((T, N),    dtype=np.float32)
            weight     = np.zeros((T,),      dtype=np.float32)

            step   = chunk_sz - overlap
            starts = list(range(0, T - overlap, step))
            # Make sure last chunk covers the end
            if starts and starts[-1] + chunk_sz < T:
                starts.append(max(0, T - chunk_sz))

            for start in starts:
                end    = min(start + chunk_sz, T)
                chunk  = frames_infer[start:end]   # already at inference resolution
                cl     = end - start

                # Remap query frame indices into this chunk's local time axis
                q_chunk = q_infer.copy()
                q_chunk[:, 0] = np.clip(q_chunk[:, 0] - start, 0, cl - 1)

                video_t   = (torch.from_numpy(chunk)
                             .permute(0, 3, 1, 2).unsqueeze(0)
                             .float().to(self.device))
                queries_t = (torch.from_numpy(q_chunk)
                             .float().unsqueeze(0).to(self.device))

                pt, pv = self.model(video_t, queries=queries_t)
                ct = pt[0].cpu().numpy()                    # (cl, N, 2)
                cv = torch.sigmoid(pv[0]).cpu().numpy()     # (cl, N)

                # Move chunk tensors off GPU immediately to free memory
                del video_t, queries_t, pt, pv
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Blend weights: full in middle, tapered at overlap edges
                w    = np.ones(cl, dtype=np.float32)
                fade = min(overlap // 2, cl // 4)
                if fade > 0:
                    w[:fade]  = np.linspace(0.5, 1.0, fade)
                    w[-fade:] = np.linspace(1.0, 0.5, fade)

                all_tracks[start:end] += ct * w[:, None, None]
                all_vis[start:end]    += cv * w[:, None]
                weight[start:end]     += w

                logger.info("CoTracker: processed frames %d-%d", start, end)

            weight     = np.maximum(weight, 1e-6)
            all_tracks /= weight[:, None, None]
            all_vis    /= weight[:, None]

        # Scale tracks back from inference resolution to original frame coords
        all_tracks[..., 0] *= scale_x
        all_tracks[..., 1] *= scale_y
        occlusions = all_vis < OCCLUSION_THRESHOLD
        return TrackResult(tracks=all_tracks, occlusions=occlusions,
                           scores=all_vis)


# =============================================================================
# Availability check (no model loaded)
# =============================================================================

def check_tapir_available() -> tuple[bool, str]:
    """Return (available, message). Does not load the model."""
    try:
        _import_tapir_model()
        import os
        ckpt_ok = bool(TAPIR_CHECKPOINT and os.path.exists(TAPIR_CHECKPOINT))
        if ckpt_ok:
            return True, "tapnet installed + checkpoint found"
        else:
            return True, (
                "tapnet installed but checkpoint missing.\n"
                f"Download tapir_checkpoint_panning.pt to:\n  {TAPIR_CHECKPOINT}"
            )
    except ImportError:
        return False, (
            "tapnet is not installed.\n\n"
            "Install with:\n"
            "  pip install git+https://github.com/google-deepmind/tapnet.git\n\n"
            "Then download the checkpoint:\n"
            "  https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt\n"
            "Place it in the checkpoints\\ folder."
        )


# =============================================================================
# Factory
# =============================================================================

def build_engine(backend: str = "auto",
                 previous_engine=None) -> object:
    """
    Load the requested tracking engine.

    backend:
        "tapir"      -- TAPIR only; raises if not installed
        "cotracker"  -- CoTracker only
        "auto"       -- tries TAPIR, falls back to CoTracker silently

    previous_engine: pass the last engine so its GPU memory is freed first.
    """
    # Explicitly release the previous engine's GPU memory before loading a new one.
    # Without this, TAPIR weights (~200 MB) stay cached on the GPU, causing
    # CoTracker to OOM when it tries to load its own model + video tensor.
    if previous_engine is not None:
        try:
            previous_engine.release()
        except Exception:
            pass
        del previous_engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    b = backend.lower().strip()

    if b in ("tapir", "auto"):
        try:
            engine = TAPIREngine()
            logger.info("Engine: TAPIR")
            return engine
        except ImportError as e:
            if b == "tapir":
                raise ImportError(
                    "tapnet is not installed.\n\n"
                    "Run this command then restart the app:\n\n"
                    "  pip install git+https://github.com/google-deepmind/tapnet.git\n\n"
                    "After installing, download the checkpoint:\n"
                    "  https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt\n"
                    "Place it in: checkpoints\\tapir_checkpoint_panning.pt"
                ) from e
            logger.warning("TAPIR unavailable (%s), falling back to CoTracker.", e)

    engine = CoTrackerEngine()
    logger.info("Engine: CoTracker")
    return engine
