from __future__ import annotations

import csv
import math
import sys
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
from tools.flow_reliability import (
    calc_forward_backward_consistency,
    make_fb_error_image,
    summarize_reliability,
)
from tools.appearance_change import (
    compute_appearance_change,
    fuse_flow_and_appearance_masks,
    save_appearance_debug_images,
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
# INPUT_PATH = r"./input_video/01-night-traffic-aerial-6749375.mp4"  # TODO: 改成你的视频路径
# OUTPUT_BASE_ROOT = Path("./flow_outputs_net_remove_test")
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

    # 方向筛选阈值缩放系数
    ANGLE_R_SCALE=1,
    ANGLE_R_MARGIN_DEG=0.0,
    ANGLE_R_MIN_DEG=3.0,
    ANGLE_R_MAX_DEG=120.0,

    # 速度大小筛选阈值缩放系数
    MAG_R_SCALE=1.8,
    MAG_R_MARGIN=0.0,
    MAG_R_MIN=0.05,
    MAG_R_MAX=200000.0,

    # 是否使用所有采样点计算速度基准 mag0
    # False：只使用方向接近主方向的点，适合作为背景速度基准
    # True ：使用全部采样点，适合方向筛选不稳定时做对比实验
    USE_ALL_POINTS_FOR_MAG0=True,

    # 最小运动幅值，小于该值的点会被过滤
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

# ========= 日志参数 =========
LOG_CFG: Dict = dict(
    ENABLE_LOGGING=True,
    LOG_TO_CONSOLE=True,
    LOG_TO_FILE=True,
    LOG_LEVEL="INFO",
    LOG_EVERY_N_FRAMES=10,
    SAVE_RUN_CONFIG_JSON=True,
)

# ========= 显示 / 输出参数 =========
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

    ENABLE_DEBUG_OUTPUTS=False,
    SAVE_DEBUG_PLOTS=True,
    SAVE_DEBUG_TABLE_CSV=False,
    SAVE_DEBUG_EXCEL_PRE=False,
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


# ========= 光流可靠性诊断参数 =========
RELIABILITY_CFG: Dict = dict(
    ENABLE_FB_CONSISTENCY=False,
    FB_ERROR_THRESHOLD=1.5,
    FB_ERROR_VIS_MAX=5.0,
    SAVE_FB_DEBUG_IMAGES=True,
    FB_SAVE_ONLY_DEBUG_FRAMES=True,
    APPLY_RELIABLE_MASK_TO_DETECTION=False,
)


# ========= 外观变化辅助检测参数 =========
APPEARANCE_CFG: Dict = dict(
    ENABLE_APPEARANCE_CHANGE=False,
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

    # ========= 第一阶段：方向筛选 =========
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

    # ========= 第二阶段：速度大小筛选 =========
    # 默认逻辑：只使用方向接近主方向的点计算速度基准 mag0。
    # 可选逻辑：如果 USE_ALL_POINTS_FOR_MAG0=True，则使用全部采样点计算速度基准 mag0。
    if inside_dir.sum() >= 30:
        use_all_points_for_mag0 = bool(alg_cfg.get("USE_ALL_POINTS_FOR_MAG0", False))

        if use_all_points_for_mag0:
            # 方案 A：使用全部采样点的速度中位数作为基准
            mag0 = float(np.median(mag))
            dmag_ref_sorted = np.sort(np.abs(mag - mag0))
            mag0_source = "all_points"
        else:
            # 方案 B：使用方向接近主方向的点的速度中位数作为基准
            mag_in = mag[inside_dir]
            mag0 = float(np.median(mag_in))
            dmag_ref_sorted = np.sort(np.abs(mag_in - mag0))
            mag0_source = "inside_direction"

        dmag = np.abs(mag - mag0)
        r0_mag, k_mag = knee_radius_1d(dmag_ref_sorted)
        r_mag = float(np.clip(
            r0_mag * alg_cfg["MAG_R_SCALE"] + alg_cfg["MAG_R_MARGIN"],
            alg_cfg["MAG_R_MIN"],
            alg_cfg["MAG_R_MAX"],
        ))

        # 速度异常只在方向接近主方向的点中判断
        mag_outlier = inside_dir & (dmag > r_mag)
    else:
        mag0 = float(np.median(mag)) if mag.size else 0.0
        r_mag = 0.0
        r0_mag, k_mag = 0.0, 0
        mag0_source = "fallback_all_points"
        mag_outlier = np.zeros_like(inside_dir, dtype=bool)

    target_s = outside_dir | mag_outlier

    # ========= 将采样点上的规则应用到整张图 =========
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
        mag0_source=mag0_source,
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
    input_path = INPUT_PATH
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]

    paths = create_run_paths(OUTPUT_BASE_ROOT, OUTPUT_RUN_PREFIX)
    if RELIABILITY_CFG["ENABLE_FB_CONSISTENCY"] and RELIABILITY_CFG["SAVE_FB_DEBUG_IMAGES"]:
        paths["flow_reliability_dir"] = paths["run_dir"] / "flow_reliability"
        paths["flow_reliability_dir"].mkdir(parents=True, exist_ok=True)
    if APPEARANCE_CFG["ENABLE_APPEARANCE_CHANGE"] and APPEARANCE_CFG["SAVE_APPEARANCE_DEBUG_IMAGES"]:
        paths["appearance_debug_dir"] = paths["run_dir"] / "appearance_debug"
        paths["appearance_debug_dir"].mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(paths, LOG_CFG) if LOG_CFG["ENABLE_LOGGING"] else None

    if logger is not None:
        logger.info("本次输出目录: %s", paths["run_dir"])
        logger.info("输入视频: %s", input_path)
        logger.info("mag0_source_mode: %s", "all_points" if ALG_CFG.get("USE_ALL_POINTS_FOR_MAG0", False) else "inside_direction")

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
            "output_base_root": str(OUTPUT_BASE_ROOT),
            "output_run_prefix": OUTPUT_RUN_PREFIX,
            "f_params": F_PARAMS,
            "alg_cfg": ALG_CFG,
            "post_cfg": POST_CFG,
            "vis_cfg": VIS_CFG,
            "reliability_cfg": RELIABILITY_CFG,
            "appearance_cfg": APPEARANCE_CFG,
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
            "frame_idx",
            "mean_speed_target",
            "mean_angle_deg_target",
            "dom_dir_deg",
            "theta_r_deg",
            "mag0",
            "mag0_source",
            "mag_r",
            "dir_outlier_pixels",
            "mag_outlier_pixels",
            "target_pixels",
            "post_target_pixels",
            "num_kept_contours",
            "mean_fb_error",
            "reliable_pixel_ratio",
            "appearance_pixels",
            "fused_target_pixels",
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

            fb_error = None
            reliable_mask = None
            mean_fb_error = 0.0
            reliable_pixel_ratio = 1.0
            if RELIABILITY_CFG["ENABLE_FB_CONSISTENCY"]:
                fb_error, reliable_mask = calc_forward_backward_consistency(prev_gray, gray_stab, flow, F_PARAMS, RELIABILITY_CFG)
                reliability_summary = summarize_reliability(fb_error, reliable_mask)
                mean_fb_error = reliability_summary["mean_fb_error"]
                reliable_pixel_ratio = reliability_summary["reliable_pixel_ratio"]

                should_save_fb = RELIABILITY_CFG["SAVE_FB_DEBUG_IMAGES"] and (
                    debug_needed if RELIABILITY_CFG["FB_SAVE_ONLY_DEBUG_FRAMES"] else True
                )
                if should_save_fb:
                    safe_imwrite(
                        paths["flow_reliability_dir"] / f"fb_error_frame_{frame_idx:05d}.png",
                        make_fb_error_image(fb_error, RELIABILITY_CFG),
                        logger=logger,
                    )
                    safe_imwrite(
                        paths["flow_reliability_dir"] / f"reliable_flow_mask_frame_{frame_idx:05d}.png",
                        reliable_mask.astype(np.uint8) * 255,
                        logger=logger,
                    )

            flow_for_detect = flow
            fx_for_detect = fx_raw
            fy_for_detect = fy_raw
            if (
                RELIABILITY_CFG["ENABLE_FB_CONSISTENCY"]
                and RELIABILITY_CFG["APPLY_RELIABLE_MASK_TO_DETECTION"]
                and reliable_mask is not None
            ):
                flow_for_detect = flow.copy()
                flow_for_detect[~reliable_mask] = 0
                fx_for_detect = flow_for_detect[..., 0]
                fy_for_detect = flow_for_detect[..., 1]

            flow_used, mask_used, info, debug_pack = two_stage_dir_then_mag(
                flow_for_detect, fx_for_detect, fy_for_detect, ALG_CFG, debug=debug_needed
            )
            if logger is not None and not info.get("ok", False):
                logger.warning("frame=%d two-stage fallback: %s", frame_idx, info.get("reason", "unknown"))

            appearance_pixels = 0
            fused_target_pixels = int(mask_used.sum())
            if APPEARANCE_CFG["ENABLE_APPEARANCE_CHANGE"]:
                appearance_pack = compute_appearance_change(prev, frame, prev_gray, gray, APPEARANCE_CFG)
                appearance_pixels = int(appearance_pack["appearance_pixels"])
                mask_used = fuse_flow_and_appearance_masks(mask_used, appearance_pack["appearance_mask"], APPEARANCE_CFG)
                fused_target_pixels = int(mask_used.sum())

                should_save_appearance = APPEARANCE_CFG["SAVE_APPEARANCE_DEBUG_IMAGES"] and (
                    debug_needed if APPEARANCE_CFG["APPEARANCE_SAVE_ONLY_DEBUG_FRAMES"] else True
                )
                if should_save_appearance:
                    save_appearance_debug_images(
                        paths["appearance_debug_dir"],
                        frame_idx,
                        appearance_pack,
                        APPEARANCE_CFG,
                    )
                    safe_imwrite(
                        paths["appearance_debug_dir"] / f"fused_candidate_mask_frame_{frame_idx:05d}.png",
                        mask_used.astype(np.uint8) * 255,
                        logger=logger,
                    )

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
                    info.get("mag0_source", "unknown"),
                    f"{info.get('r_mag', 0.0):.6f}",
                    int(info.get("dir_outlier_pixels", 0)),
                    int(info.get("mag_outlier_pixels", 0)),
                    int(info.get("target_pixels", 0)),
                    int(post_pack["post_target_pixels"]),
                    int(post_pack["num_kept_contours"]),
                    f"{mean_fb_error:.6f}",
                    f"{reliable_pixel_ratio:.6f}",
                    int(appearance_pixels),
                    int(fused_target_pixels),
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
                    "frame=%d mean_speed=%.3f mean_angle=%.2f dom_dir=%.2f r_theta=%.2f mag0=%.3f mag0_source=%s r_mag=%.3f target_pixels=%d appearance_pixels=%d fused_pixels=%d post_pixels=%d kept_regions=%d mean_fb_error=%.3f reliable_ratio=%.3f",
                    frame_idx,
                    speed,
                    angle,
                    float(info.get("dom_dir_deg", 0.0)),
                    float(info.get("r_theta", 0.0)),
                    float(info.get("mag0", 0.0)),
                    info.get("mag0_source", "unknown"),
                    float(info.get("r_mag", 0.0)),
                    int(info.get("target_pixels", 0)),
                    int(appearance_pixels),
                    int(fused_target_pixels),
                    int(post_pack["post_target_pixels"]),
                    int(post_pack["num_kept_contours"]),
                    mean_fb_error,
                    reliable_pixel_ratio,
                )

            prev = frame
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
