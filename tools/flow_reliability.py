from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple


def _cv2():
    import cv2

    return cv2


def _np():
    import numpy as np

    return np


DEFAULT_F_PARAMS: Dict = dict(
    pyr_scale=0.5,
    levels=4,
    winsize=25,
    iterations=3,
    poly_n=7,
    poly_sigma=1.5,
    flags=None,
)

DEFAULT_RELIABILITY_CFG: Dict = dict(
    FB_ERROR_THRESHOLD=1.5,
    FB_ERROR_VIS_MAX=5.0,
)


def calc_forward_flow(prev_gray, gray, f_params: Dict | None = None):
    params = f_params or DEFAULT_F_PARAMS
    cv2 = _cv2()
    flags = params["flags"] if params.get("flags") is not None else cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    return cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=gray,
        flow=None,
        pyr_scale=params["pyr_scale"],
        levels=params["levels"],
        winsize=params["winsize"],
        iterations=params["iterations"],
        poly_n=params["poly_n"],
        poly_sigma=params["poly_sigma"],
        flags=flags,
    )


def calc_forward_backward_consistency(
    prev_gray,
    gray,
    forward_flow,
    f_params: Dict | None = None,
    reliability_cfg: Dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    np = _np()
    params = f_params or DEFAULT_F_PARAMS
    cfg = reliability_cfg or DEFAULT_RELIABILITY_CFG
    cv2 = _cv2()
    flags = params["flags"] if params.get("flags") is not None else cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    backward_flow = cv2.calcOpticalFlowFarneback(
        prev=gray,
        next=prev_gray,
        flow=None,
        pyr_scale=params["pyr_scale"],
        levels=params["levels"],
        winsize=params["winsize"],
        iterations=params["iterations"],
        poly_n=params["poly_n"],
        poly_sigma=params["poly_sigma"],
        flags=flags,
    )

    h, w = prev_gray.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs + forward_flow[..., 0].astype(np.float32)
    map_y = ys + forward_flow[..., 1].astype(np.float32)

    valid = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    bx = cv2.remap(
        backward_flow[..., 0],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    by = cv2.remap(
        backward_flow[..., 1],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    fb_x = forward_flow[..., 0] + bx
    fb_y = forward_flow[..., 1] + by
    error = np.sqrt(fb_x * fb_x + fb_y * fb_y)
    error[~valid] = np.inf

    threshold = float(cfg["FB_ERROR_THRESHOLD"])
    reliable_mask = np.isfinite(error) & (error <= threshold)
    return error, reliable_mask


def make_fb_error_image(error, reliability_cfg: Dict | None = None):
    cv2 = _cv2()
    np = _np()
    cfg = reliability_cfg or DEFAULT_RELIABILITY_CFG
    vis_max = max(float(cfg["FB_ERROR_VIS_MAX"]), 1e-6)
    err = np.where(np.isfinite(error), error, vis_max)
    err_u8 = np.clip(err / vis_max * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(err_u8, cv2.COLORMAP_JET)


def summarize_reliability(error, reliable_mask) -> Dict[str, float]:
    np = _np()
    finite_error = error[np.isfinite(error)]
    mean_error = float(np.mean(finite_error)) if finite_error.size > 0 else 0.0
    reliable_ratio = float(np.mean(reliable_mask)) if reliable_mask is not None else 1.0
    return dict(mean_fb_error=mean_error, reliable_pixel_ratio=reliable_ratio)


def save_reliability_images(out_dir: str | Path, frame_idx: int, error, reliable_mask, reliability_cfg: Dict | None = None):
    cv2 = _cv2()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    error_path = out_dir / f"fb_error_frame_{frame_idx:05d}.png"
    mask_path = out_dir / f"reliable_flow_mask_frame_{frame_idx:05d}.png"
    cv2.imwrite(str(error_path), make_fb_error_image(error, reliability_cfg))
    cv2.imwrite(str(mask_path), reliable_mask.astype(np.uint8) * 255)
    return error_path, mask_path


def run_video_reliability_diagnosis(
    input_path: str | Path,
    output_dir: str | Path,
    debug_interval: int = 10,
    f_params: Dict | None = None,
    reliability_cfg: Dict | None = None,
):
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2 = _cv2()
    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"VideoCapture open failed: {input_path}")

    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Could not read first frame: {input_path}")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    stats_path = output_dir / "flow_reliability_stats.csv"
    frame_idx = 0

    with open(stats_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame_idx", "mean_fb_error", "reliable_pixel_ratio"])

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = calc_forward_flow(prev_gray, gray, f_params)
                error, reliable_mask = calc_forward_backward_consistency(
                    prev_gray,
                    gray,
                    flow,
                    f_params,
                    reliability_cfg,
                )
                summary = summarize_reliability(error, reliable_mask)
                writer.writerow([
                    frame_idx,
                    f"{summary['mean_fb_error']:.6f}",
                    f"{summary['reliable_pixel_ratio']:.6f}",
                ])

                if frame_idx % max(1, int(debug_interval)) == 0:
                    save_reliability_images(output_dir, frame_idx, error, reliable_mask, reliability_cfg)

                prev_gray = gray
                frame_idx += 1
        finally:
            cap.release()

    return stats_path


def main():
    parser = argparse.ArgumentParser(description="Diagnose optical-flow reliability with forward-backward consistency.")
    parser.add_argument("input_video", help="Input video path.")
    parser.add_argument("--output-dir", default="flow_reliability_outputs", help="Folder for reliability images and CSV.")
    parser.add_argument("--debug-interval", type=int, default=10, help="Save reliability images every N frames.")
    parser.add_argument("--fb-error-threshold", type=float, default=1.5, help="Maximum error considered reliable.")
    parser.add_argument("--fb-error-vis-max", type=float, default=5.0, help="Maximum error value for heatmap visualization.")
    args = parser.parse_args()

    reliability_cfg = dict(
        FB_ERROR_THRESHOLD=args.fb_error_threshold,
        FB_ERROR_VIS_MAX=args.fb_error_vis_max,
    )
    stats_path = run_video_reliability_diagnosis(
        args.input_video,
        args.output_dir,
        debug_interval=args.debug_interval,
        reliability_cfg=reliability_cfg,
    )
    print(f"[OK] Reliability diagnosis saved: {stats_path}")


if __name__ == "__main__":
    main()
