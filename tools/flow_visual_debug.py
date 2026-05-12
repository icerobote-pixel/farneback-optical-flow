from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .save_manager import safe_imwrite


def draw_quiver(frame_bgr, flow, step=16, scale=2.0, show_text=False, mag_text_th=0.15):
    h, w = frame_bgr.shape[:2]
    vis = frame_bgr.copy()

    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step].astype(np.int32)
    fx = flow[ys, xs, 0]
    fy = flow[ys, xs, 1]

    pts0 = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    pts1 = np.stack([xs + (fx * scale).astype(np.int32), ys + (fy * scale).astype(np.int32)], axis=-1).reshape(-1, 2)

    for (x0, y0), (x1, y1), dx, dy in zip(pts0, pts1, fx.reshape(-1), fy.reshape(-1)):
        cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 1, tipLength=0.3)
        if show_text:
            mag = float(np.sqrt(dx * dx + dy * dy))
            if mag >= mag_text_th:
                ang = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
                cv2.putText(
                    vis,
                    f"{ang:.0f}|{mag:.2f}",
                    (int(x0) + 2, int(y0) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.33,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
    return vis


def flow_to_hsv(flow):
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    h = (ang / 2).astype(np.uint8)
    s = np.full_like(h, 255, dtype=np.uint8)
    max_mag = float(mag.max()) if np.isfinite(mag).any() and float(mag.max()) > 0 else 1.0
    v = np.clip((mag / (max_mag + 1e-6) * 255), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def mean_flow_stats(flow):
    fx = flow[..., 0]
    fy = flow[..., 1]
    valid = np.isfinite(fx) & np.isfinite(fy) & ((fx != 0) | (fy != 0))
    if not valid.any():
        return 0.0, 0.0
    mean_fx = float(np.mean(fx[valid]))
    mean_fy = float(np.mean(fy[valid]))
    speed = float(np.sqrt(mean_fx ** 2 + mean_fy ** 2))
    angle_deg = (math.degrees(math.atan2(mean_fy, mean_fx)) + 360.0) % 360.0
    return speed, angle_deg


def bool_mask_to_u8(mask_bool):
    if mask_bool is None:
        return None
    return mask_bool.astype(np.uint8) * 255


def overlay_mask(frame_bgr, mask_bool, color=(0, 0, 255), alpha=0.35):
    vis = frame_bgr.copy()
    if mask_bool is None:
        return vis
    mask_bool = mask_bool.astype(bool)
    if not mask_bool.any():
        return vis
    color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_layer[:] = color
    vis[mask_bool] = cv2.addWeighted(frame_bgr[mask_bool], 1.0 - alpha, color_layer[mask_bool], alpha, 0)
    return vis


def make_region_id_map(shape_hw: Tuple[int, int], contours):
    rid = np.zeros(shape_hw, dtype=np.int32)
    for idx, cnt in enumerate(contours, start=1):
        cv2.drawContours(rid, [cnt], -1, idx, thickness=-1)
    return rid


def postprocess_target_mask(mask_bool, post_cfg: Dict):
    if mask_bool is None:
        return dict(
            pre_mask_bool=None,
            morph_mask_bool=None,
            post_mask_bool=None,
            pre_mask_u8=None,
            morph_mask_u8=None,
            post_mask_u8=None,
            raw_contours=[],
            kept_contours=[],
            kept_boxes=[],
            region_id_map=None,
            num_raw_contours=0,
            num_kept_contours=0,
            pre_target_pixels=0,
            morph_target_pixels=0,
            post_target_pixels=0,
        )

    pre_mask_bool = mask_bool.astype(bool)
    pre_mask_u8 = bool_mask_to_u8(pre_mask_bool)

    if post_cfg["ENABLE_MASK_MORPHOLOGY"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, post_cfg["MORPH_KERNEL_SIZE"])
        morph_mask_u8 = pre_mask_u8.copy()
        if post_cfg["MORPH_OPEN_ITER"] > 0:
            morph_mask_u8 = cv2.morphologyEx(morph_mask_u8, cv2.MORPH_OPEN, kernel, iterations=post_cfg["MORPH_OPEN_ITER"])
        if post_cfg["MORPH_CLOSE_ITER"] > 0:
            morph_mask_u8 = cv2.morphologyEx(morph_mask_u8, cv2.MORPH_CLOSE, kernel, iterations=post_cfg["MORPH_CLOSE_ITER"])
    else:
        morph_mask_u8 = pre_mask_u8.copy()

    morph_mask_bool = morph_mask_u8 > 0
    contours, _ = cv2.findContours(morph_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    post_mask_u8 = np.zeros_like(morph_mask_u8)
    kept_contours = []
    kept_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if post_cfg["ENABLE_AREA_FILTER"] and (area < post_cfg["MIN_REGION_AREA"] or area > post_cfg["MAX_REGION_AREA"]):
            continue
        kept_contours.append(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        kept_boxes.append((x, y, bw, bh))
        cv2.drawContours(post_mask_u8, [cnt], -1, 255, thickness=-1)

    post_mask_bool = post_mask_u8 > 0
    region_id_map = make_region_id_map(post_mask_bool.shape, kept_contours)

    return dict(
        pre_mask_bool=pre_mask_bool,
        morph_mask_bool=morph_mask_bool,
        post_mask_bool=post_mask_bool,
        pre_mask_u8=pre_mask_u8,
        morph_mask_u8=morph_mask_u8,
        post_mask_u8=post_mask_u8,
        raw_contours=contours,
        kept_contours=kept_contours,
        kept_boxes=kept_boxes,
        region_id_map=region_id_map,
        num_raw_contours=len(contours),
        num_kept_contours=len(kept_contours),
        pre_target_pixels=int(pre_mask_bool.sum()),
        morph_target_pixels=int(morph_mask_bool.sum()),
        post_target_pixels=int(post_mask_bool.sum()),
    )


def draw_post_frames(frame_bgr, post_pack):
    region_frame = frame_bgr.copy()
    box_frame = frame_bgr.copy()
    combined_frame = frame_bgr.copy()

    for cnt, (x, y, bw, bh) in zip(post_pack["kept_contours"], post_pack["kept_boxes"]):
        cv2.drawContours(region_frame, [cnt], -1, (0, 0, 255), 2)
        cv2.drawContours(combined_frame, [cnt], -1, (0, 0, 255), 2)

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        text = f"cx={int(cx)}, cy={int(cy)}, w={bw}, h={bh}"

        cv2.rectangle(box_frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
        cv2.rectangle(combined_frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
        cv2.circle(box_frame, (int(cx), int(cy)), 3, (255, 0, 0), -1)
        cv2.circle(combined_frame, (int(cx), int(cy)), 3, (255, 0, 0), -1)

        ty = max(0, y - 5)
        pos = (x, ty if ty > 0 else y + 15)
        cv2.putText(box_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combined_frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return region_frame, box_frame, combined_frame


def save_debug(debug_pack, frame_idx: int, vis_cfg: Dict, paths: Dict, logger=None):
    if not vis_cfg["SAVE_DEBUG_PLOTS"] and not vis_cfg["SAVE_DEBUG_TABLE_CSV"]:
        return

    try:
        pts = debug_pack["pts"]
        mag = debug_pack["mag"]
        theta = debug_pack["theta"]
        xs = debug_pack["xs"]
        ys = debug_pack["ys"]
        dom_dir = debug_pack["dom_dir"]
        hist = debug_pack["hist"]
        dtheta_deg = debug_pack["dtheta_deg"]
        r_theta = debug_pack["r_theta"]
        mag0 = debug_pack["mag0"]
        r_mag = debug_pack["r_mag"]
        inside_dir = debug_pack["inside_dir"].astype(bool)
        target_s = debug_pack["target_s"].astype(bool)
        out_dir = paths["cluster_plot_dir"]

        if vis_cfg["SAVE_DEBUG_PLOTS"]:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            ax.scatter(pts[~target_s, 0], pts[~target_s, 1], s=4, alpha=0.25, label="background-like")
            ax.scatter(pts[target_s, 0], pts[target_s, 1], s=7, alpha=0.75, label="target")
            ax.set_xlabel("fx")
            ax.set_ylabel("fy")
            ax.set_title(f"Frame {frame_idx} | two-stage target vs bg")
            ax.grid(True)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(str(out_dir / f"scatter_fx_fy_frame_{frame_idx:05d}.png"), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.2, 3.2))
            ax.plot(hist, linewidth=1.5)
            ax.set_title(f"Frame {frame_idx} angle histogram")
            ax.set_xlabel("angle bin")
            ax.set_ylabel("count")
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(str(out_dir / f"angle_hist_frame_{frame_idx:05d}.png"), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            ax.scatter(dtheta_deg[~target_s], mag[~target_s], s=4, alpha=0.25, label="bg-like")
            ax.scatter(dtheta_deg[target_s], mag[target_s], s=7, alpha=0.75, label="target")
            ax.axvline(r_theta, linestyle="--", linewidth=1, label="theta radius")
            ax.axhline(mag0, linestyle="--", linewidth=1, label="mag0")
            ax.set_xlabel("angle deviation (deg)")
            ax.set_ylabel("magnitude (px/frame)")
            ax.grid(True)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(str(out_dir / f"angle_mag_frame_{frame_idx:05d}.png"), dpi=150)
            plt.close(fig)

        if vis_cfg["SAVE_DEBUG_TABLE_CSV"]:
            table_path = out_dir / f"table_frame_{frame_idx:05d}.csv"
            angle_deg = (np.rad2deg(theta) + 360.0) % 360.0
            dom_dir_deg = (np.rad2deg(dom_dir) + 360.0) % 360.0
            abs_dmag = np.abs(mag - mag0)
            with open(table_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "sx", "sy", "fx", "fy", "mag", "angle_deg", "dom_dir_deg",
                    "abs_dtheta_deg", "r_theta_deg", "inside_dir", "mag0", "abs_dmag", "r_mag", "is_target",
                ])
                for i in range(pts.shape[0]):
                    writer.writerow([
                        int(xs[i]), int(ys[i]), float(pts[i, 0]), float(pts[i, 1]), float(mag[i]),
                        float(angle_deg[i]), float(dom_dir_deg), float(dtheta_deg[i]), float(r_theta),
                        int(inside_dir[i]), float(mag0), float(abs_dmag[i]), float(r_mag), int(target_s[i]),
                    ])
    except Exception as exc:
        if logger is not None:
            logger.warning("Debug output failed for frame %d: %s", frame_idx, exc)
        else:
            print(f"[WARN] Debug output failed for frame {frame_idx}: {exc}")


def save_debug_excel_prepost(debug_pack, post_pack, frame_idx: int, vis_cfg: Dict, paths: Dict, logger=None):
    if (not vis_cfg["SAVE_DEBUG_EXCEL_PRE"]) and (not vis_cfg["SAVE_COMPARE_EXCEL"]):
        return

    try:
        import pandas as pd

        pts = debug_pack["pts"]
        mag = debug_pack["mag"]
        theta = debug_pack["theta"]
        xs = debug_pack["xs"]
        ys = debug_pack["ys"]
        dtheta_deg = debug_pack["dtheta_deg"]
        target_s = debug_pack["target_s"].astype(bool)
        inside_dir = debug_pack["inside_dir"].astype(bool)
        mag_outlier = debug_pack["mag_outlier"].astype(bool)

        sample_df = pd.DataFrame(
            {
                "x": xs,
                "y": ys,
                "fx": pts[:, 0],
                "fy": pts[:, 1],
                "magnitude": mag,
                "angle_deg": (np.rad2deg(theta) + 360.0) % 360.0,
                "angle_deviation_deg": dtheta_deg,
                "inside_direction": inside_dir.astype(int),
                "magnitude_outlier": mag_outlier.astype(int),
                "is_target": target_s.astype(int),
            }
        )
        summary_df = pd.DataFrame(
            [
                ["frame_idx", frame_idx],
                ["pre_target_pixels", int(post_pack["pre_target_pixels"])],
                ["morph_target_pixels", int(post_pack["morph_target_pixels"])],
                ["post_target_pixels", int(post_pack["post_target_pixels"])],
                ["num_raw_contours", int(post_pack["num_raw_contours"])],
                ["num_kept_contours", int(post_pack["num_kept_contours"])],
            ],
            columns=["item", "value"],
        )

        out_xlsx = paths["excel_dir"] / f"grid_frame_{frame_idx:05d}.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            sample_df.to_excel(writer, sheet_name="sample_points", index=False)

        if logger is not None:
            logger.info("Excel debug workbook saved: %s", out_xlsx)
        else:
            print("[OK] Excel debug workbook saved:", out_xlsx)
    except Exception as exc:
        if logger is not None:
            logger.warning("Excel output failed for frame %d: %s", frame_idx, exc)
        else:
            print(f"[WARN] Excel output failed for frame {frame_idx}: {exc}")


def save_compare_outputs(frame, frame_idx: int, post_pack: Dict, region_frame, box_frame, combined_frame, vis_cfg: Dict, paths: Dict, logger=None):
    pre_mask_bool = post_pack["pre_mask_bool"]
    post_mask_bool = post_pack["post_mask_bool"]

    if vis_cfg["SAVE_COMPARE_PRE_MASK_IMAGE"]:
        safe_imwrite(paths["compare_pre_dir"] / f"mask_pre_frame_{frame_idx:05d}.png", post_pack["pre_mask_u8"], logger=logger)
    if vis_cfg["SAVE_COMPARE_PRE_OVERLAY_IMAGE"]:
        safe_imwrite(paths["compare_pre_dir"] / f"overlay_pre_frame_{frame_idx:05d}.png", overlay_mask(frame, pre_mask_bool, color=(0, 0, 255), alpha=0.35), logger=logger)
    if vis_cfg["SAVE_COMPARE_POST_MORPH_MASK_IMAGE"]:
        safe_imwrite(paths["compare_post_dir"] / f"mask_post_morph_frame_{frame_idx:05d}.png", post_pack["morph_mask_u8"], logger=logger)
    if vis_cfg["SAVE_COMPARE_POST_FINAL_MASK_IMAGE"]:
        safe_imwrite(paths["compare_post_dir"] / f"mask_post_final_frame_{frame_idx:05d}.png", post_pack["post_mask_u8"], logger=logger)
    if vis_cfg["SAVE_COMPARE_POST_OVERLAY_IMAGE"]:
        safe_imwrite(paths["compare_post_dir"] / f"overlay_post_frame_{frame_idx:05d}.png", overlay_mask(frame, post_mask_bool, color=(0, 255, 0), alpha=0.35), logger=logger)
    if vis_cfg["SAVE_COMPARE_REGION_IMAGE"]:
        safe_imwrite(paths["compare_post_dir"] / f"region_post_frame_{frame_idx:05d}.png", region_frame, logger=logger)
    if vis_cfg["SAVE_COMPARE_BOX_IMAGE"]:
        safe_imwrite(paths["compare_post_dir"] / f"box_post_frame_{frame_idx:05d}.png", box_frame, logger=logger)
    if vis_cfg["SAVE_COMPARE_COMBINED_IMAGE"]:
        safe_imwrite(paths["compare_post_dir"] / f"combined_post_frame_{frame_idx:05d}.png", combined_frame, logger=logger)


def save_raw_speed_hist(all_mags, paths: Dict, vis_cfg: Dict, logger=None):
    if len(all_mags) == 0:
        return
    all_mags = np.concatenate(all_mags, axis=0)
    all_mags = all_mags[np.isfinite(all_mags)]
    if all_mags.size == 0:
        return

    max_mag = float(all_mags.max()) if all_mags.max() > 0 else 1.0
    bins = np.linspace(0, max_mag, 201)
    hist, edges = np.histogram(all_mags, bins=bins)
    centers_mag = 0.5 * (edges[:-1] + edges[1:])

    if vis_cfg["SAVE_RAW_SPEED_HIST_CSV"]:
        with open(paths["raw_speed_hist_csv"], "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["speed_bin_center_px_per_frame", "pixel_count"])
            for speed, count in zip(centers_mag, hist):
                writer.writerow([f"{speed:.6f}", int(count)])

    if vis_cfg["SAVE_RAW_SPEED_HIST_PNG"]:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 4))
            plt.plot(centers_mag, hist)
            plt.xlabel("Speed (pixels/frame)")
            plt.ylabel("Number of pixels")
            plt.title("Distribution of optical flow magnitude (RAW)")
            plt.tight_layout()
            plt.savefig(paths["raw_speed_hist_png"], dpi=150)
            plt.close()
            if logger is not None:
                logger.info("Raw speed histogram saved: %s", paths["raw_speed_hist_png"])
        except Exception as exc:
            if logger is not None:
                logger.warning("Could not save raw speed histogram PNG: %s", exc)
            else:
                print("[WARN] Could not save raw speed histogram PNG:", exc)
