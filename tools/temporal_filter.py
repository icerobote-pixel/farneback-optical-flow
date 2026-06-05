from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict

import cv2
import numpy as np


DEFAULT_TEMPORAL_CFG: Dict = dict(
    ENABLE_TEMPORAL_FILTER=True,
    HISTORY_LENGTH=5,
    MIN_HIT_FRAMES=3,
    MIN_HISTORY_FRAMES=3,
    MOTION_TOLERANCE_DILATE_ITER=1,
    KEEP_CURRENT_ONLY=True,
    WARMUP_MODE="passthrough",
    SAVE_TEMPORAL_DEBUG_IMAGES=True,
    TEMPORAL_SAVE_ONLY_DEBUG_FRAMES=True,
)


def bool_to_u8(mask_bool):
    return mask_bool.astype(np.uint8) * 255


class TemporalMaskFilter:
    """Filter candidate masks using recent-frame voting."""

    def __init__(self, cfg: Dict | None = None):
        self.cfg = {**DEFAULT_TEMPORAL_CFG, **(cfg or {})}
        self._validate_cfg()
        self.history = deque(maxlen=int(self.cfg["HISTORY_LENGTH"]))
        self.shape = None

    def _validate_cfg(self):
        history_length = int(self.cfg["HISTORY_LENGTH"])
        min_hit_frames = int(self.cfg["MIN_HIT_FRAMES"])
        min_history_frames = int(self.cfg["MIN_HISTORY_FRAMES"])
        if history_length < 1:
            raise ValueError("HISTORY_LENGTH must be at least 1")
        if not 1 <= min_hit_frames <= history_length:
            raise ValueError("MIN_HIT_FRAMES must be between 1 and HISTORY_LENGTH")
        if not 1 <= min_history_frames <= history_length:
            raise ValueError("MIN_HISTORY_FRAMES must be between 1 and HISTORY_LENGTH")
        if self.cfg["WARMUP_MODE"] not in {"passthrough", "suppress", "adaptive_vote"}:
            raise ValueError("WARMUP_MODE must be passthrough, suppress, or adaptive_vote")

    def reset(self):
        self.history.clear()
        self.shape = None

    def _make_history_mask(self, current_mask):
        iterations = int(self.cfg["MOTION_TOLERANCE_DILATE_ITER"])
        if iterations <= 0:
            return current_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(bool_to_u8(current_mask), kernel, iterations=iterations) > 0

    def update(self, candidate_mask):
        current_mask = np.asarray(candidate_mask, dtype=bool)
        if current_mask.ndim != 2:
            raise ValueError("candidate_mask must be a 2D mask")
        if self.shape is not None and current_mask.shape != self.shape:
            self.reset()
        self.shape = current_mask.shape

        self.history.append(self._make_history_mask(current_mask))
        history_stack = np.stack(tuple(self.history), axis=0)
        hit_count = history_stack.sum(axis=0, dtype=np.uint16)
        frames_in_history = len(self.history)
        ready = frames_in_history >= int(self.cfg["MIN_HISTORY_FRAMES"])

        if ready:
            temporal_mask = hit_count >= int(self.cfg["MIN_HIT_FRAMES"])
        elif self.cfg["WARMUP_MODE"] == "passthrough":
            temporal_mask = current_mask.copy()
        elif self.cfg["WARMUP_MODE"] == "suppress":
            temporal_mask = np.zeros(current_mask.shape, dtype=bool)
        else:
            adaptive_hits = min(int(self.cfg["MIN_HIT_FRAMES"]), frames_in_history)
            temporal_mask = hit_count >= adaptive_hits

        if self.cfg["KEEP_CURRENT_ONLY"]:
            temporal_mask &= current_mask

        confidence = hit_count.astype(np.float32) / float(frames_in_history)
        return dict(
            current_mask=current_mask,
            temporal_mask=temporal_mask,
            hit_count=hit_count,
            confidence=confidence,
            frames_in_history=frames_in_history,
            temporal_ready=ready,
            input_pixels=int(current_mask.sum()),
            temporal_pixels=int(temporal_mask.sum()),
        )


def save_temporal_debug_images(out_dir: str | Path, frame_idx: int, pack: Dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"temporal_input_mask_frame_{frame_idx:05d}.png"), bool_to_u8(pack["current_mask"]))
    cv2.imwrite(str(out_dir / f"temporal_output_mask_frame_{frame_idx:05d}.png"), bool_to_u8(pack["temporal_mask"]))

    history_size = max(1, int(pack["frames_in_history"]))
    hit_vis = np.clip(pack["hit_count"].astype(np.float32) / history_size * 255.0, 0, 255).astype(np.uint8)
    confidence_vis = np.clip(pack["confidence"] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"temporal_hit_count_frame_{frame_idx:05d}.png"), hit_vis)
    cv2.imwrite(
        str(out_dir / f"temporal_confidence_frame_{frame_idx:05d}.png"),
        cv2.applyColorMap(confidence_vis, cv2.COLORMAP_JET),
    )
