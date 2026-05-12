from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from tools.flow_visual_debug import (
    draw_post_frames,
    draw_quiver,
    flow_to_hsv,
    mean_flow_stats,
    postprocess_target_mask,
    save_compare_outputs,
    save_debug,
    save_debug_excel_prepost,
    save_raw_speed_hist,
)
from tools.save_manager import (
    create_run_paths,
    open_requested_writers,
    release_writers,
    safe_imwrite,
    save_run_config,
    setup_run_logger,
)


# ========= 输入 / 输出 =========
INPUT_PATH = r"./input_video/cam2.mp4"  # TODO: 改成你的视频路径
OUTPUT_BASE_ROOT = Path("./flow_outputs_net_remove4")
OUTPUT_RUN_PREFIX = "flow_outputs_two_stage_dir_mag_compare"

# ========= 光流参数 =========
F_PARAMS = dict(
    pyr_scale=0.5,
    levels=4,
    winsize=25,
    iterations=3,
    poly_n=7,
    poly_sigma=1.5,
    flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
)

# ========= 采样与两阶段判别参数 =========
ALG_CFG: Dict = dict(
    CLUSTER_USE_ALL_POINTS=False,
    CLUSTER_SAMPLE_STEP=4,
    MIN_MAG_FOR_POINTS=0.0,
    MIN_CLUSTER_POINTS=1000,
    ANGLE_HIST_BINS=180,
    DOM_DIR_REFINE_DEG=12.0,
    ANGLE_R_SCALE=1.0,
    ANGLE_R_MARGIN_DEG=0.0,
    ANGLE_R_MIN_DEG=3.0,
    ANGLE_R_MAX_DEG=120.0,
    MAG_R_SCALE=1.8,
    MAG_R_MARGIN=0.0,
    MAG_R_MIN=0.05,
    MAG_R_MAX=200000.0,
    FISH_MAG_MIN=0.1,
)

# ========= 后处理参数 =========
POST_CFG: Dict = dict(
    ENABLE_MASK_MORPHOLOGY=True,
    MORPH_KERNEL_SIZE=(3, 3),
    MORPH_OPEN_ITER=1,
    MORPH_CLOSE_ITER=1,
    ENABLE_AREA_FILTER=True,
    MIN_REGION_AREA=800,
    MAX_REGION_AREA=50000,
)

# ========= 显示 / 输出参数 =========

# ========= 日志参数 =========
LOG_CFG: Dict = dict(
    ENABLE_LOGGING=True,
    LOG_TO_CONSOLE=True,
    LOG_TO_FILE=True,
    LOG_LEVEL="INFO",
    LOG_EVERY_N_FRAMES=10,
    SAVE_RUN_CONFIG_JSON=True,
)

VIS_CFG: Dict = dict(
    DEBUG_INTERVAL=10,
    COMPARE_SAVE_ONLY_DEBUG_FRAMES=True,
    QUIVER_STEP=16,
    QUIVER_SCALE=5.0,
    ENABLE_GLOBAL_STAB=False,

    SAVE_VIDEO_HSV=True,
    SAVE_VIDEO_QUIVER=True,
    SAVE_VIDEO_REGION=True,
    SAVE_VIDEO_BOX=True,
    SAVE_VIDEO_COMBINED=True,

    ENABLE_DEBUG_OUTPUTS=True,
    SAVE_DEBUG_PLOTS=True,
    SAVE_DEBUG_TABLE_CSV=True,
    SAVE_DEBUG_EXCEL_PRE=True,
    SAVE_DEBUG_DETECT_FRAME=True,
    SAVE_DEBUG_QUIVER_RAW=True,

    ENABLE_COMPARE_OUTPUTS=True,
    SAVE_COMPARE_PRE_MASK_IMAGE=True,
    SAVE_COMPARE_PRE_OVERLAY_IMAGE=True,
    SAVE_COMPARE_POST_MORPH_MASK_IMAGE=True,
    SAVE_COMPARE_POST_FINAL_MASK_IMAGE=True,
    SAVE_COMPARE_POST_OVERLAY_IMAGE=True,
    SAVE_COMPARE_REGION_IMAGE=True,
    SAVE_COMPARE_BOX_IMAGE=True,
    SAVE_COMPARE_COMBINED_IMAGE=True,
    SAVE_COMPARE_EXCEL=True,

    EXCEL_SAVE_TEXT_GRID=True,
    EXCEL_SAVE_MAG_GRID=True,
    EXCEL_SAVE_ANGLE_GRID=True,
    EXCEL_SAVE_DTHETA_GRID=True,
    EXCEL_SAVE_DMAG_GRID=True,
    EXCEL_SAVE_TARGET_GRID=True,
    EXCEL_SAVE_INSIDE_DIR_GRID=True,
    EXCEL_ENABLE_COLOR=True,
    EXCEL_HIGHLIGHT_TARGET=True,
    EXCEL_HIGHLIGHT_INSIDE_DIR=True,
    EXCEL_TEXT_SHOW_ANGLE_MAG=True,

    SAVE_STATS_CSV=True,
    SAVE_RAW_SPEED_HIST_CSV=True,
    SAVE_RAW_SPEED_HIST_PNG=True,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect moving regions in a video with Farneback optical flow.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=INPUT_PATH,
        help=f"Input video path. Default: {INPUT_PATH}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(OUTPUT_BASE_ROOT),
        help=f"Output root directory. Default: {OUTPUT_BASE_ROOT}",
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=VIS_CFG["DEBUG_INTERVAL"],
        help="Save debug/compare outputs every N frames.",
    )
    parser.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip Excel debug output for faster runs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write logs to file without printing progress to the console.",
    )
    return parser.parse_args()


def wrap_angle_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circular_mean(theta):
    s = np.mean(np.sin(theta))
    c = np.mean(np.cos(theta))
    return float(np.arctan2(s, c))


def deg2rad(d):
    return d * np.pi / 180.0


def rad2deg(r):
    return r * 180.0 / np.pi


def knee_radius_1d(sorted_vals):
    if sorted_vals.size < 20:
        r0 = float(sorted_vals[-1]) if sorted_vals.size > 0 else 0.0
        return r0, 0

    y = sorted_vals.astype(np.float64)
    n = y.size
    x = np.linspace(0.0, 1.0, n)
    y_norm = y / (y[-1] + 1e-12)

    x0, y0 = 0.0, y_norm[0]
    x1, y1 = 1.0, y_norm[-1]
    vx, vy = (x1 - x0), (y1 - y0)
    denom = np.sqrt(vx * vx + vy * vy) + 1e-12

    px = x - x0
    py = y_norm - y0
    dist = np.abs(px * vy - py * vx) / denom
    k = int(np.argmax(dist))
    return float(y[k]), k


def dominant_direction_hist(theta, mag, alg_cfg: Dict):
    th = (theta + 2 * np.pi) % (2 * np.pi)
    bins = alg_cfg["ANGLE_HIST_BINS"]
    idx = np.floor(th / (2 * np.pi) * bins).astype(np.int32)
    idx = np.clip(idx, 0, bins - 1)
    hist = np.bincount(idx, minlength=bins)

    best = int(np.argmax(hist))
    center = (best + 0.5) / bins * 2 * np.pi
    center = wrap_angle_pi(center)

    refine_r = deg2rad(alg_cfg["DOM_DIR_REFINE_DEG"])
    d = np.abs(wrap_angle_pi(theta - center))
    mask = d <= refine_r
    if mask.sum() >= 30:
        center = circular_mean(theta[mask])
    return float(center), hist


def sample_flow_points(fx_raw, fy_raw, alg_cfg: Dict):
    h, w = fx_raw.shape[:2]
    if alg_cfg["CLUSTER_USE_ALL_POINTS"]:
        fx = fx_raw
        fy = fy_raw
        ys, xs = np.mgrid[0:h, 0:w]
    else:
        step = alg_cfg["CLUSTER_SAMPLE_STEP"]
        fx = fx_raw[::step, ::step]
        fy = fy_raw[::step, ::step]
        ys, xs = np.mgrid[0:fx.shape[0], 0:fx.shape[1]]
        ys = ys * step
        xs = xs * step

    fxv = fx.reshape(-1)
    fyv = fy.reshape(-1)
    xsv = xs.reshape(-1).astype(np.int32)
    ysv = ys.reshape(-1).astype(np.int32)

    pts = np.stack([fxv, fyv], axis=1)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    xsv = xsv[finite]
    ysv = ysv[finite]

    mag = np.linalg.norm(pts, axis=1)
    keep = mag >= alg_cfg["MIN_MAG_FOR_POINTS"]
    pts = pts[keep]
    xsv = xsv[keep]
    ysv = ysv[keep]
    return pts.astype(np.float32), xsv, ysv


def estimate_global_affine(prev_gray, gray):
    pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=8)
    if pts_prev is None:
        return np.eye(2, 3, dtype=np.float32)

    pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts_prev, None)
    if pts_next is None:
        return np.eye(2, 3, dtype=np.float32)

    ok = status.flatten() == 1
    if ok.sum() < 10:
        return np.eye(2, 3, dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(
        pts_prev[ok], pts_next[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    return M.astype(np.float32) if M is not None else np.eye(2, 3, dtype=np.float32)


def two_stage_dir_then_mag(flow, fx_raw, fy_raw, alg_cfg: Dict, debug=False):
    pts, xs, ys = sample_flow_points(fx_raw, fy_raw, alg_cfg)
    if pts.shape[0] < alg_cfg["MIN_CLUSTER_POINTS"]:
        mask = np.ones_like(fx_raw, dtype=bool)
        return flow, mask, {"ok": False, "reason": "too_few_points"}, None

    fx = pts[:, 0]
    fy = pts[:, 1]
    mag = np.sqrt(fx * fx + fy * fy)
    theta = np.arctan2(fy, fx)

    dom_dir, hist = dominant_direction_hist(theta, mag, alg_cfg)
    dtheta = np.abs(wrap_angle_pi(theta - dom_dir))
    dtheta_deg = rad2deg(dtheta)

    dtheta_sorted = np.sort(dtheta_deg)
    r0_theta, k_theta = knee_radius_1d(dtheta_sorted)
    r_theta = float(np.clip(
        r0_theta * alg_cfg["ANGLE_R_SCALE"] + alg_cfg["ANGLE_R_MARGIN_DEG"],
        alg_cfg["ANGLE_R_MIN_DEG"],
        alg_cfg["ANGLE_R_MAX_DEG"],
    ))

    inside_dir = dtheta_deg <= r_theta
    outside_dir = ~inside_dir

    if inside_dir.sum() >= 30:
        mag_in = mag[inside_dir]
        mag0 = float(np.median(mag_in))
        dmag = np.abs(mag - mag0)
        dmag_in_sorted = np.sort(np.abs(mag_in - mag0))
        r0_mag, k_mag = knee_radius_1d(dmag_in_sorted)
        r_mag = float(np.clip(
            r0_mag * alg_cfg["MAG_R_SCALE"] + alg_cfg["MAG_R_MARGIN"],
            alg_cfg["MAG_R_MIN"],
            alg_cfg["MAG_R_MAX"],
        ))
        mag_outlier = inside_dir & (dmag > r_mag)
    else:
        mag0 = float(np.median(mag)) if mag.size else 0.0
        r_mag = 0.0
        r0_mag, k_mag = 0.0, 0
        mag_outlier = np.zeros_like(inside_dir, dtype=bool)

    target_s = outside_dir | mag_outlier

    mag_full = np.sqrt(fx_raw * fx_raw + fy_raw * fy_raw)
    theta_full = np.arctan2(fy_raw, fx_raw)
    dtheta_full_deg = np.abs(rad2deg(wrap_angle_pi(theta_full - dom_dir)))
    inside_dir_full = dtheta_full_deg <= r_theta
    dmag_full = np.abs(mag_full - mag0)
    mag_outlier_full = inside_dir_full & (dmag_full > r_mag)

    mask_target_full = (~inside_dir_full) | mag_outlier_full
    mask_target_full = mask_target_full & (mag_full >= alg_cfg["FISH_MAG_MIN"])

    flow_used = np.zeros_like(flow, dtype=flow.dtype)
    flow_used[..., 0][mask_target_full] = fx_raw[mask_target_full]
    flow_used[..., 1][mask_target_full] = fy_raw[mask_target_full]

    info = dict(
        ok=True,
        dom_dir_rad=float(dom_dir),
        dom_dir_deg=float((rad2deg(dom_dir) + 360.0) % 360.0),
        r_theta=float(r_theta),
        r0_theta=float(r0_theta),
        r_mag=float(r_mag),
        r0_mag=float(r0_mag),
        mag0=float(mag0),
        n=int(pts.shape[0]),
        target_pixels=int(mask_target_full.sum()),
        inside_dir_pixels=int(inside_dir_full.sum()),
        mag_outlier_pixels=int(mag_outlier_full.sum()),
        dir_outlier_pixels=int((~inside_dir_full).sum()),
    )

    debug_pack = None
    if debug:
        debug_pack = dict(
            pts=pts,
            xs=xs,
            ys=ys,
            mag=mag,
            theta=theta,
            dom_dir=dom_dir,
            hist=hist,
            dtheta_deg=dtheta_deg,
            r_theta=r_theta,
            r0_theta=r0_theta,
            k_theta=k_theta,
            mag0=mag0,
            r_mag=r_mag,
            r0_mag=r0_mag,
            k_mag=k_mag,
            inside_dir=inside_dir.astype(np.int32),
            mag_outlier=mag_outlier.astype(np.int32),
            target_s=target_s.astype(np.int32),
        )

    return flow_used, mask_target_full, info, debug_pack


def main():
    args = parse_args()
    input_path = args.input
    output_base_root = Path(args.output_dir)

    VIS_CFG["DEBUG_INTERVAL"] = max(1, int(args.debug_interval))
    if args.no_excel:
        VIS_CFG["SAVE_DEBUG_EXCEL_PRE"] = False
        VIS_CFG["SAVE_COMPARE_EXCEL"] = False
    if args.quiet:
        LOG_CFG["LOG_TO_CONSOLE"] = False

    paths = create_run_paths(output_base_root, OUTPUT_RUN_PREFIX)
    logger = setup_run_logger(paths, LOG_CFG) if LOG_CFG["ENABLE_LOGGING"] else None

    if logger is not None:
        logger.info("本次输出目录: %s", paths["run_dir"])
        logger.info("输入视频: %s", input_path)

    cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        if logger is not None:
            logger.error("VideoCapture 打开失败: %s", input_path)
        raise RuntimeError("VideoCapture 打开失败，请检查路径/FFmpeg。")

    ret, prev = cap.read()
    if not ret:
        if logger is not None:
            logger.error("无法读取第一帧: %s", input_path)
        raise RuntimeError("无法读取第一帧。")

    h, w = prev.shape[:2]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    fps_read = cap.get(cv2.CAP_PROP_FPS)
    fps = max(10, int(fps_read) if fps_read and not np.isnan(fps_read) else 10)

    if logger is not None:
        logger.info("视频信息: width=%d height=%d fps_read=%.3f fps_used=%d", w, h, float(fps_read or 0.0), fps)

    if LOG_CFG["SAVE_RUN_CONFIG_JSON"]:
        run_config = {
            "input_path": str(input_path),
            "output_base_root": str(output_base_root),
            "output_run_prefix": OUTPUT_RUN_PREFIX,
            "f_params": F_PARAMS,
            "alg_cfg": ALG_CFG,
            "post_cfg": POST_CFG,
            "vis_cfg": VIS_CFG,
            "log_cfg": LOG_CFG,
            "video_width": int(w),
            "video_height": int(h),
            "fps_read": float(fps_read or 0.0),
            "fps_used": int(fps),
        }
        save_run_config(paths, run_config, logger=logger)

    video_flags = {
        "hsv": VIS_CFG["SAVE_VIDEO_HSV"],
        "quiver": VIS_CFG["SAVE_VIDEO_QUIVER"],
        "region": VIS_CFG["SAVE_VIDEO_REGION"],
        "box": VIS_CFG["SAVE_VIDEO_BOX"],
        "combined": VIS_CFG["SAVE_VIDEO_COMBINED"],
    }
    writers, writer_paths = open_requested_writers(paths, video_flags, fps, (w, h), logger=logger)
    if writer_paths and logger is not None:
        for key, p in writer_paths.items():
            logger.info("视频输出 %s: %s", key, p)

    all_mags = []
    need_debug_pack = VIS_CFG["ENABLE_DEBUG_OUTPUTS"] or (VIS_CFG["ENABLE_COMPARE_OUTPUTS"] and VIS_CFG["SAVE_COMPARE_EXCEL"])

    csv_file = None
    stats_writer = None
    if VIS_CFG["SAVE_STATS_CSV"]:
        csv_file = open(paths["stats_csv"], "w", newline="", encoding="utf-8")
        stats_writer = csv.writer(csv_file)
        stats_writer.writerow([
            "frame_idx", "mean_speed_target", "mean_angle_deg_target", "dom_dir_deg", "theta_r_deg",
            "mag0", "mag_r", "dir_outlier_pixels", "mag_outlier_pixels", "target_pixels",
            "post_target_pixels", "num_kept_contours",
        ])

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if VIS_CFG["ENABLE_GLOBAL_STAB"]:
                M = estimate_global_affine(prev_gray, gray)
                gray_stab = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR)
            else:
                gray_stab = gray

            flow = cv2.calcOpticalFlowFarneback(
                prev=prev_gray,
                next=gray_stab,
                flow=None,
                pyr_scale=F_PARAMS["pyr_scale"],
                levels=F_PARAMS["levels"],
                winsize=F_PARAMS["winsize"],
                iterations=F_PARAMS["iterations"],
                poly_n=F_PARAMS["poly_n"],
                poly_sigma=F_PARAMS["poly_sigma"],
                flags=F_PARAMS["flags"],
            )

            fx_raw = flow[..., 0]
            fy_raw = flow[..., 1]
            mag_raw = np.sqrt(fx_raw ** 2 + fy_raw ** 2)
            mag_flat = mag_raw.reshape(-1)
            mag_flat = mag_flat[np.isfinite(mag_flat)]
            if mag_flat.size > 0:
                all_mags.append(mag_flat)

            debug_needed = need_debug_pack and (frame_idx % VIS_CFG["DEBUG_INTERVAL"] == 0)
            compare_needed = VIS_CFG["ENABLE_COMPARE_OUTPUTS"] and (
                (frame_idx % VIS_CFG["DEBUG_INTERVAL"] == 0) if VIS_CFG["COMPARE_SAVE_ONLY_DEBUG_FRAMES"] else True
            )

            flow_used, mask_used, info, debug_pack = two_stage_dir_then_mag(
                flow, fx_raw, fy_raw, ALG_CFG, debug=debug_needed
            )
            if logger is not None and not info.get("ok", False):
                logger.warning("frame=%d two-stage fallback: %s", frame_idx, info.get("reason", "unknown"))

            post_pack = postprocess_target_mask(mask_used, POST_CFG)
            flow_hsv_bgr = flow_to_hsv(flow_used)
            quiver_img = draw_quiver(frame, flow_used, step=VIS_CFG["QUIVER_STEP"], scale=VIS_CFG["QUIVER_SCALE"])

            quiver_debug_img = None
            if debug_needed and VIS_CFG["SAVE_DEBUG_QUIVER_RAW"]:
                quiver_debug_img = draw_quiver(
                    frame, flow, step=VIS_CFG["QUIVER_STEP"], scale=VIS_CFG["QUIVER_SCALE"], show_text=False
                )

            speed, angle = mean_flow_stats(flow_used)
            cv2.putText(
                quiver_img,
                f"Two-stage: dir->mag | mean_target: {speed:.2f}px/f angle:{angle:.1f}deg",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                quiver_img,
                f"dom_dir={info.get('dom_dir_deg', 0):.1f}  r_theta={info.get('r_theta', 0):.1f}deg  mag0={info.get('mag0', 0):.2f}  r_mag={info.get('r_mag', 0):.2f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
            )

            region_frame, box_frame, combined_frame = draw_post_frames(frame, post_pack)

            if "hsv" in writers:
                writers["hsv"].write(flow_hsv_bgr)
            if "quiver" in writers:
                writers["quiver"].write(quiver_img)
            if "region" in writers:
                writers["region"].write(region_frame)
            if "box" in writers:
                writers["box"].write(box_frame)
            if "combined" in writers:
                writers["combined"].write(combined_frame)

            if stats_writer is not None:
                stats_writer.writerow([
                    frame_idx,
                    f"{speed:.6f}",
                    f"{angle:.3f}",
                    f"{info.get('dom_dir_deg', 0.0):.3f}",
                    f"{info.get('r_theta', 0.0):.3f}",
                    f"{info.get('mag0', 0.0):.6f}",
                    f"{info.get('r_mag', 0.0):.6f}",
                    int(info.get("dir_outlier_pixels", 0)),
                    int(info.get("mag_outlier_pixels", 0)),
                    int(info.get("target_pixels", 0)),
                    int(post_pack["post_target_pixels"]),
                    int(post_pack["num_kept_contours"]),
                ])

            if VIS_CFG["ENABLE_DEBUG_OUTPUTS"] and debug_needed and debug_pack is not None and info.get("ok", False):
                save_debug(debug_pack, frame_idx, VIS_CFG, paths, logger=logger)
                if VIS_CFG["SAVE_DEBUG_EXCEL_PRE"] or VIS_CFG["SAVE_COMPARE_EXCEL"]:
                    save_debug_excel_prepost(debug_pack, post_pack, frame_idx, VIS_CFG, paths, logger=logger)
                if VIS_CFG["SAVE_DEBUG_DETECT_FRAME"]:
                    safe_imwrite(paths["detect_frame_dir"] / f"detect_frame_{frame_idx:05d}.png", combined_frame, logger=logger)
                if VIS_CFG["SAVE_DEBUG_QUIVER_RAW"] and quiver_debug_img is not None:
                    safe_imwrite(paths["quiver_debug_dir"] / f"quiver_frame_{frame_idx:05d}.png", quiver_debug_img, logger=logger)

            if compare_needed:
                save_compare_outputs(frame, frame_idx, post_pack, region_frame, box_frame, combined_frame, VIS_CFG, paths, logger=logger)


            if logger is not None and (frame_idx % max(1, LOG_CFG["LOG_EVERY_N_FRAMES"]) == 0):
                logger.info(
                    "frame=%d mean_speed=%.3f mean_angle=%.2f dom_dir=%.2f r_theta=%.2f mag0=%.3f r_mag=%.3f target_pixels=%d post_pixels=%d kept_regions=%d",
                    frame_idx,
                    speed,
                    angle,
                    float(info.get("dom_dir_deg", 0.0)),
                    float(info.get("r_theta", 0.0)),
                    float(info.get("mag0", 0.0)),
                    float(info.get("r_mag", 0.0)),
                    int(info.get("target_pixels", 0)),
                    int(post_pack["post_target_pixels"]),
                    int(post_pack["num_kept_contours"]),
                )

            prev_gray = gray_stab
            frame_idx += 1
    finally:
        cap.release()
        release_writers(writers, logger=logger)
        if csv_file is not None:
            csv_file.close()

    save_raw_speed_hist(all_mags, paths, VIS_CFG, logger=logger)
    if logger is not None:
        logger.info("处理完成。输出位于: %s", paths["run_dir"].resolve())
    else:
        print("[OK] 处理完成。输出位于：", paths["run_dir"].resolve())


if __name__ == "__main__":
    main()
