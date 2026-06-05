from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np


DEFAULT_APPEARANCE_CFG: Dict = dict(
    ENABLE_APPEARANCE_CHANGE=True,
    FUSION_MODE="flow_or_appearance",

    ENABLE_COLOR_CHANGE=True,
    COLOR_BLUR_SIZE=(5, 5),
    COLOR_DIFF_THRESHOLD=22.0,
    COLOR_USE_HS_ONLY=True,

    ENABLE_EDGE_CHANGE=True,
    EDGE_CANNY_LOW=40,
    EDGE_CANNY_HIGH=120,
    EDGE_DILATE_ITER=1,
    EDGE_DIFF_DILATE_ITER=1,

    ENABLE_TEXTURE_CHANGE=True,
    TEXTURE_BLUR_SIZE=(3, 3),
    TEXTURE_DIFF_THRESHOLD=18.0,

    ENABLE_APPEARANCE_MORPH=True,
    APPEARANCE_KERNEL_SIZE=(3, 3),
    APPEARANCE_OPEN_ITER=0,
    APPEARANCE_CLOSE_ITER=1,

    SAVE_APPEARANCE_DEBUG_IMAGES=True,
    APPEARANCE_SAVE_ONLY_DEBUG_FRAMES=True,
)


def _odd_kernel(size):
    if isinstance(size, int):
        size = (size, size)
    kx, ky = int(size[0]), int(size[1])
    kx = max(1, kx + (kx % 2 == 0))
    ky = max(1, ky + (ky % 2 == 0))
    return kx, ky


def bool_to_u8(mask_bool):
    return mask_bool.astype(np.uint8) * 255


def compute_color_change_mask(prev_bgr, curr_bgr, cfg: Dict):
    prev_blur = cv2.GaussianBlur(prev_bgr, _odd_kernel(cfg["COLOR_BLUR_SIZE"]), 0)
    curr_blur = cv2.GaussianBlur(curr_bgr, _odd_kernel(cfg["COLOR_BLUR_SIZE"]), 0)
    prev_hsv = cv2.cvtColor(prev_blur, cv2.COLOR_BGR2HSV).astype(np.float32)
    curr_hsv = cv2.cvtColor(curr_blur, cv2.COLOR_BGR2HSV).astype(np.float32)

    if cfg["COLOR_USE_HS_ONLY"]:
        dh = np.abs(prev_hsv[..., 0] - curr_hsv[..., 0])
        dh = np.minimum(dh, 180.0 - dh)
        ds = np.abs(prev_hsv[..., 1] - curr_hsv[..., 1])
        diff = np.sqrt(dh * dh + ds * ds)
    else:
        dh = np.abs(prev_hsv[..., 0] - curr_hsv[..., 0])
        dh = np.minimum(dh, 180.0 - dh)
        ds = np.abs(prev_hsv[..., 1] - curr_hsv[..., 1])
        dv = np.abs(prev_hsv[..., 2] - curr_hsv[..., 2])
        diff = np.sqrt(dh * dh + ds * ds + dv * dv)

    mask = diff >= float(cfg["COLOR_DIFF_THRESHOLD"])
    return mask, diff


def compute_edge_change_mask(prev_gray, curr_gray, cfg: Dict):
    prev_edges = cv2.Canny(prev_gray, int(cfg["EDGE_CANNY_LOW"]), int(cfg["EDGE_CANNY_HIGH"]))
    curr_edges = cv2.Canny(curr_gray, int(cfg["EDGE_CANNY_LOW"]), int(cfg["EDGE_CANNY_HIGH"]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if cfg["EDGE_DILATE_ITER"] > 0:
        prev_edges = cv2.dilate(prev_edges, kernel, iterations=int(cfg["EDGE_DILATE_ITER"]))
        curr_edges = cv2.dilate(curr_edges, kernel, iterations=int(cfg["EDGE_DILATE_ITER"]))

    diff = cv2.absdiff(prev_edges, curr_edges)
    if cfg["EDGE_DIFF_DILATE_ITER"] > 0:
        diff = cv2.dilate(diff, kernel, iterations=int(cfg["EDGE_DIFF_DILATE_ITER"]))
    return diff > 0, diff


def compute_texture_change_mask(prev_gray, curr_gray, cfg: Dict):
    prev_blur = cv2.GaussianBlur(prev_gray, _odd_kernel(cfg["TEXTURE_BLUR_SIZE"]), 0)
    curr_blur = cv2.GaussianBlur(curr_gray, _odd_kernel(cfg["TEXTURE_BLUR_SIZE"]), 0)
    prev_lap = cv2.Laplacian(prev_blur, cv2.CV_32F, ksize=3)
    curr_lap = cv2.Laplacian(curr_blur, cv2.CV_32F, ksize=3)
    diff = np.abs(curr_lap - prev_lap)
    mask = diff >= float(cfg["TEXTURE_DIFF_THRESHOLD"])
    return mask, diff


def postprocess_appearance_mask(mask_bool, cfg: Dict):
    if not cfg["ENABLE_APPEARANCE_MORPH"]:
        return mask_bool.astype(bool)

    mask_u8 = bool_to_u8(mask_bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg["APPEARANCE_KERNEL_SIZE"])
    if cfg["APPEARANCE_OPEN_ITER"] > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=int(cfg["APPEARANCE_OPEN_ITER"]))
    if cfg["APPEARANCE_CLOSE_ITER"] > 0:
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=int(cfg["APPEARANCE_CLOSE_ITER"]))
    return mask_u8 > 0


def compute_appearance_change(prev_bgr, curr_bgr, prev_gray, curr_gray, cfg: Dict | None = None):
    cfg = {**DEFAULT_APPEARANCE_CFG, **(cfg or {})}
    h, w = curr_gray.shape[:2]
    zero_mask = np.zeros((h, w), dtype=bool)

    color_mask, color_diff = (zero_mask, np.zeros((h, w), dtype=np.float32))
    edge_mask, edge_diff = (zero_mask, np.zeros((h, w), dtype=np.uint8))
    texture_mask, texture_diff = (zero_mask, np.zeros((h, w), dtype=np.float32))

    if cfg["ENABLE_COLOR_CHANGE"]:
        color_mask, color_diff = compute_color_change_mask(prev_bgr, curr_bgr, cfg)
    if cfg["ENABLE_EDGE_CHANGE"]:
        edge_mask, edge_diff = compute_edge_change_mask(prev_gray, curr_gray, cfg)
    if cfg["ENABLE_TEXTURE_CHANGE"]:
        texture_mask, texture_diff = compute_texture_change_mask(prev_gray, curr_gray, cfg)

    raw_mask = color_mask | edge_mask | texture_mask
    final_mask = postprocess_appearance_mask(raw_mask, cfg)
    return dict(
        color_mask=color_mask,
        edge_mask=edge_mask,
        texture_mask=texture_mask,
        appearance_raw_mask=raw_mask,
        appearance_mask=final_mask,
        color_diff=color_diff,
        edge_diff=edge_diff,
        texture_diff=texture_diff,
        appearance_pixels=int(final_mask.sum()),
    )


def fuse_flow_and_appearance_masks(flow_mask, appearance_mask, cfg: Dict):
    mode = cfg.get("FUSION_MODE", "flow_or_appearance")
    flow_mask = flow_mask.astype(bool)
    appearance_mask = appearance_mask.astype(bool)
    if mode == "flow_only":
        return flow_mask
    if mode == "appearance_only":
        return appearance_mask
    if mode == "flow_and_appearance":
        return flow_mask & appearance_mask
    if mode == "flow_or_appearance":
        return flow_mask | appearance_mask
    raise ValueError(f"Unknown appearance fusion mode: {mode}")


def _diff_to_u8(diff, vis_max=None):
    diff = np.asarray(diff, dtype=np.float32)
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return np.zeros(diff.shape, dtype=np.uint8)
    if vis_max is None:
        vis_max = max(float(np.percentile(finite, 99)), 1e-6)
    return np.clip(diff / vis_max * 255.0, 0, 255).astype(np.uint8)


def save_appearance_debug_images(out_dir: str | Path, frame_idx: int, pack: Dict, cfg: Dict):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"color_change_mask_frame_{frame_idx:05d}.png"), bool_to_u8(pack["color_mask"]))
    cv2.imwrite(str(out_dir / f"edge_change_mask_frame_{frame_idx:05d}.png"), bool_to_u8(pack["edge_mask"]))
    cv2.imwrite(str(out_dir / f"texture_change_mask_frame_{frame_idx:05d}.png"), bool_to_u8(pack["texture_mask"]))
    cv2.imwrite(str(out_dir / f"appearance_change_mask_frame_{frame_idx:05d}.png"), bool_to_u8(pack["appearance_mask"]))

    color_vis = cv2.applyColorMap(_diff_to_u8(pack["color_diff"]), cv2.COLORMAP_JET)
    texture_vis = cv2.applyColorMap(_diff_to_u8(pack["texture_diff"]), cv2.COLORMAP_JET)
    cv2.imwrite(str(out_dir / f"color_diff_frame_{frame_idx:05d}.png"), color_vis)
    cv2.imwrite(str(out_dir / f"texture_diff_frame_{frame_idx:05d}.png"), texture_vis)
