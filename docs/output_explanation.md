# Output Explanation

This document explains the output folders and files created by `main_flow_detect.py`.

Each run creates a numbered folder under `OUTPUT_BASE_ROOT`, for example:

```text
flow_outputs_net_remove4/
`-- flow_outputs_two_stage_dir_mag_compare_run_001/
```

The exact folder name depends on `OUTPUT_BASE_ROOT` and `OUTPUT_RUN_PREFIX` in `main_flow_detect.py`.

## Run Folder Overview

| Path | Meaning |
| --- | --- |
| `videos/` | Result videos for different visual views. |
| `stats/` | Frame-level statistics and raw speed histogram files. |
| `cluster_plots/` | Algorithm diagnostic plots and per-sample CSV tables. |
| `frame_detections/` | Saved final detection images for debug frames. |
| `quiver_debug/` | Raw optical-flow arrow images for debug frames. |
| `compare_pre/` | Images before morphology and area filtering. |
| `compare_post/` | Images after morphology and area filtering. |
| `excel/` | Excel debug workbooks for selected frames. |
| `run.log` | Log file for the run. |
| `run_config.json` | Saved copy of the configuration used in the run. |

## `videos/`

These files are controlled by the `SAVE_VIDEO_*` switches in `VIS_CFG`.

| File | Meaning |
| --- | --- |
| `flow_hsv_two_stage.mp4` or `.avi` | HSV optical-flow visualization. Hue represents motion direction; brightness represents motion magnitude. |
| `flow_quiver_two_stage.mp4` or `.avi` | Arrow visualization of the filtered target flow. Each arrow shows local motion direction and relative size. |
| `flow_regions_two_stage.mp4` or `.avi` | Original video with detected target contours drawn. |
| `flow_boxes_two_stage.mp4` or `.avi` | Original video with bounding boxes, center points, width, and height. |
| `flow_regions_boxes_two_stage.mp4` or `.avi` | Combined view with contours, boxes, center points, and size text. |

The writer tries MP4 first. If MP4 initialization fails, it falls back to AVI.

## `stats/`

### `flow_stats_two_stage.csv`

This CSV stores one row per processed frame.

| Column | Meaning |
| --- | --- |
| `frame_idx` | Frame number. |
| `mean_speed_target` | Mean speed of the filtered target flow, in pixels per frame. |
| `mean_angle_deg_target` | Mean direction angle of the filtered target flow, in degrees. |
| `dom_dir_deg` | Dominant motion direction estimated from sampled raw flow points. |
| `theta_r_deg` | Direction radius threshold. Points outside this radius are direction outliers. |
| `mag0` | Baseline speed used for magnitude outlier detection. |
| `mag0_source` | Source used for `mag0`: `all_points`, `inside_direction`, or fallback. |
| `mag_r` | Magnitude radius threshold. Points farther than this from `mag0` are magnitude outliers. |
| `dir_outlier_pixels` | Number of full-frame pixels outside the dominant direction radius. |
| `mag_outlier_pixels` | Number of full-frame pixels inside the dominant direction but abnormal in speed. |
| `target_pixels` | Number of pixels in the raw target mask before post-processing. |
| `post_target_pixels` | Number of pixels in the final target mask after post-processing. |
| `num_kept_contours` | Number of detected regions kept after area filtering. |

### `flow_speed_hist_raw.csv`

This CSV describes the raw optical-flow speed distribution over the whole processed video.

| Column | Meaning |
| --- | --- |
| `speed_bin_center_px_per_frame` | Center of a speed bin, in pixels per frame. |
| `pixel_count` | Number of pixels whose raw flow magnitude falls into that speed bin. |

### `flow_speed_hist_raw.png`

A chart version of `flow_speed_hist_raw.csv`. It helps show whether most motion is small background movement or large motion.

## `cluster_plots/`

These files are created for debug frames only. The interval is controlled by `DEBUG_INTERVAL`.

| File Pattern | Meaning |
| --- | --- |
| `scatter_fx_fy_frame_XXXXX.png` | Scatter plot of optical-flow vectors. The x-axis is `fx`, the y-axis is `fy`. It shows background-like points and target points. |
| `angle_hist_frame_XXXXX.png` | Histogram used to estimate the dominant motion direction. The tallest bin usually represents the main background/camera motion direction. |
| `angle_mag_frame_XXXXX.png` | Plot of angle deviation versus magnitude. It shows how direction and speed thresholds separate target-like points. |
| `table_frame_XXXXX.csv` | Per-sample debug table for the frame. It contains sampled positions, flow vectors, magnitudes, angles, thresholds, and target flags. |

### `table_frame_XXXXX.csv` Columns

| Column | Meaning |
| --- | --- |
| `sx`, `sy` | Sample point position in the frame. |
| `fx`, `fy` | Optical-flow vector components. |
| `mag` | Flow magnitude, in pixels per frame. |
| `angle_deg` | Motion direction angle of this sample point. |
| `dom_dir_deg` | Dominant motion direction for this frame. |
| `abs_dtheta_deg` | Absolute angular difference from the dominant direction. |
| `r_theta_deg` | Direction threshold radius. |
| `inside_dir` | `1` if the point is inside the dominant direction range; otherwise `0`. |
| `mag0` | Baseline magnitude for this frame. |
| `abs_dmag` | Absolute difference between this point's magnitude and `mag0`. |
| `r_mag` | Magnitude threshold radius. |
| `is_target` | `1` if the sampled point is classified as target-like; otherwise `0`. |

## `frame_detections/`

| File Pattern | Meaning |
| --- | --- |
| `detect_frame_XXXXX.png` | Final combined detection image for a debug frame. It contains contours, boxes, center points, and size text. |

## `quiver_debug/`

| File Pattern | Meaning |
| --- | --- |
| `quiver_frame_XXXXX.png` | Raw optical-flow arrows before target filtering. This is useful for checking whether the original flow field looks reasonable. |

## `compare_pre/`

These images show what the algorithm produced before morphology and area filtering.

| File Pattern | Meaning |
| --- | --- |
| `mask_pre_frame_XXXXX.png` | Raw target mask. White pixels are detected as target; black pixels are background. |
| `overlay_pre_frame_XXXXX.png` | Original frame with the raw target mask overlaid in red. |

## `compare_post/`

These images show what remains after post-processing.

| File Pattern | Meaning |
| --- | --- |
| `mask_post_morph_frame_XXXXX.png` | Mask after morphology. Opening removes small noise; closing fills small holes and connects nearby regions. |
| `mask_post_final_frame_XXXXX.png` | Final mask after morphology and area filtering. |
| `overlay_post_frame_XXXXX.png` | Original frame with the final mask overlaid in green. |
| `region_post_frame_XXXXX.png` | Original frame with final target contours. |
| `box_post_frame_XXXXX.png` | Original frame with final bounding boxes, center points, width, and height. |
| `combined_post_frame_XXXXX.png` | Combined final view with contours and boxes together. |

## `excel/`

Excel workbooks are created for debug frames when `SAVE_DEBUG_EXCEL_PRE` or `SAVE_COMPARE_EXCEL` is enabled.

File pattern:

```text
grid_frame_XXXXX.xlsx
```

Current implementation note: despite the `grid_` name, the current workbook contains two sheets: `summary` and `sample_points`.

### Sheet: `summary`

This sheet gives frame-level post-processing statistics.

| Item | Meaning |
| --- | --- |
| `frame_idx` | Frame number. |
| `pre_target_pixels` | Number of target pixels before post-processing. |
| `morph_target_pixels` | Number of target pixels after morphology. |
| `post_target_pixels` | Number of target pixels after morphology and area filtering. |
| `num_raw_contours` | Number of connected regions found after morphology. |
| `num_kept_contours` | Number of regions kept after area filtering. |

### Sheet: `sample_points`

This sheet stores one row per sampled optical-flow point.

| Column | Meaning |
| --- | --- |
| `x`, `y` | Sample point position in the frame. |
| `fx`, `fy` | Optical-flow vector components. Positive `fx` means movement to the right; positive `fy` means movement downward in image coordinates. |
| `magnitude` | Flow speed, in pixels per frame. |
| `angle_deg` | Movement direction angle, in degrees. |
| `angle_deviation_deg` | Absolute direction difference from the dominant direction. |
| `inside_direction` | `1` if the point is close to the dominant direction; otherwise `0`. |
| `magnitude_outlier` | `1` if the point is inside the dominant direction but has abnormal speed; otherwise `0`. |
| `is_target` | `1` if the sample point is classified as target-like; otherwise `0`. |

## `run.log`

The log records run progress, video information, output paths, threshold values, target pixel counts, and kept region counts. It is useful for checking whether the run behaved normally without opening every image.

## `run_config.json`

This file stores the exact configuration used for the run, including:

- input video path
- output folder settings
- Farneback optical-flow parameters
- algorithm parameters
- post-processing parameters
- visualization/output switches
- logging settings
- video width, height, and FPS

Keep this file when comparing different parameter experiments. It makes each output folder reproducible.

## `appearance_debug/`

This Version 2.0 folder is created when appearance-change detection and its
debug-image output are enabled.

| File Pattern | Meaning |
| --- | --- |
| `color_change_mask_frame_XXXXX.png` | Pixels with significant HSV color change. |
| `edge_change_mask_frame_XXXXX.png` | Pixels with changed Canny edge structure. |
| `texture_change_mask_frame_XXXXX.png` | Pixels with changed Laplacian texture response. |
| `appearance_change_mask_frame_XXXXX.png` | Combined and cleaned appearance-change mask. |
| `color_diff_frame_XXXXX.png` | Color-difference heatmap. |
| `texture_diff_frame_XXXXX.png` | Texture-difference heatmap. |
| `fused_candidate_mask_frame_XXXXX.png` | Final candidate mask after flow/appearance fusion. |

The frame statistics CSV and `run.log` also include `appearance_pixels` and
`fused_target_pixels`.

## `flow_reliability/`

This folder contains optional forward-backward optical-flow consistency
diagnostics. The statistics CSV and log include `mean_fb_error` and
`reliable_pixel_ratio`.

## `temporal_debug/`

This Version 3.0 folder is created by `main_flow_detect_temporal.py` when
temporal debug output is enabled.

| File Pattern | Meaning |
| --- | --- |
| `temporal_input_mask_frame_XXXXX.png` | Candidate mask before temporal filtering. |
| `temporal_output_mask_frame_XXXXX.png` | Candidate mask retained by recent-frame voting. |
| `temporal_hit_count_frame_XXXXX.png` | Grayscale visualization of recent-frame hit counts. |
| `temporal_confidence_frame_XXXXX.png` | Heatmap of the fraction of recent masks supporting each pixel. |

The statistics CSV and `run.log` also include `temporal_target_pixels`,
`temporal_history_frames`, and `temporal_ready`.
