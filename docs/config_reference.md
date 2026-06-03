# Configuration Reference

This document explains the optional features and tuning switches in `main_flow_detect.py`.

## Input And Output

| Name | Meaning |
| --- | --- |
| `INPUT_PATH` | Default video path used when no command-line video path is provided. |
| `OUTPUT_BASE_ROOT` | Base folder where run output folders are created. |
| `OUTPUT_RUN_PREFIX` | Prefix used for numbered run folders, such as `prefix_run_001`. |

## Farneback Optical Flow Parameters

`F_PARAMS` controls OpenCV Farneback optical flow. These values affect the raw motion field before target filtering.

| Name | Meaning |
| --- | --- |
| `pyr_scale` | Image pyramid scale. Smaller values use a stronger pyramid reduction. |
| `levels` | Number of pyramid levels. More levels can handle larger motion but may be slower. |
| `winsize` | Averaging window size. Larger windows are smoother but may blur small motion. |
| `iterations` | Iterations at each pyramid level. More iterations can improve stability but cost time. |
| `poly_n` | Pixel neighborhood size for polynomial expansion. |
| `poly_sigma` | Gaussian standard deviation for polynomial expansion. |
| `flags` | OpenCV Farneback flags. The current code uses Gaussian filtering. |

## Algorithm Configuration: `ALG_CFG`

These settings control the two-stage target decision: first direction filtering, then magnitude filtering.

| Name | Meaning |
| --- | --- |
| `CLUSTER_USE_ALL_POINTS` | If `True`, use every pixel for direction and speed analysis. If `False`, use sampled pixels. |
| `CLUSTER_SAMPLE_STEP` | Sampling interval used when `CLUSTER_USE_ALL_POINTS=False`. For example, `4` means one sample every 4 pixels. |
| `MIN_MAG_FOR_POINTS` | Minimum raw optical-flow magnitude required for a sample point to participate in analysis. |
| `MIN_CLUSTER_POINTS` | Minimum number of valid sample points. If fewer points are available, the algorithm uses a fallback path. |
| `ANGLE_HIST_BINS` | Number of bins in the direction histogram used to estimate the dominant motion direction. |
| `DOM_DIR_REFINE_DEG` | Angular neighborhood used to refine the dominant direction after histogram selection. |
| `ANGLE_R_SCALE` | Scale applied to the automatically estimated direction radius. |
| `ANGLE_R_MARGIN_DEG` | Extra margin added to the direction radius. |
| `ANGLE_R_MIN_DEG` | Lower bound for the direction radius. |
| `ANGLE_R_MAX_DEG` | Upper bound for the direction radius. |
| `MAG_R_SCALE` | Scale applied to the automatically estimated magnitude radius. |
| `MAG_R_MARGIN` | Extra margin added to the magnitude radius. |
| `MAG_R_MIN` | Lower bound for the magnitude radius. |
| `MAG_R_MAX` | Upper bound for the magnitude radius. |
| `USE_ALL_POINTS_FOR_MAG0` | If `True`, compute the baseline speed `mag0` from all sample points. If `False`, compute it from points near the dominant direction. |
| `FISH_MAG_MIN` | Final minimum movement magnitude. Pixels below this value are removed from the target mask. |

### How The Two-Stage Decision Works

1. The program samples optical-flow vectors from the frame.
2. It estimates the dominant motion direction from a direction histogram.
3. Points far from that dominant direction are treated as direction outliers.
4. It computes a baseline speed `mag0`.
5. Points near the dominant direction but with abnormal speed are treated as magnitude outliers.
6. The target mask is built from direction outliers plus magnitude outliers, then filtered by `FISH_MAG_MIN`.

## Post-Processing Configuration: `POST_CFG`

These settings clean the raw target mask after algorithm detection.

| Name | Meaning |
| --- | --- |
| `ENABLE_MASK_MORPHOLOGY` | Enables morphology processing to remove noise and connect broken target regions. |
| `MORPH_KERNEL_SIZE` | Kernel size used for morphology operations. Larger kernels create stronger cleanup. |
| `MORPH_OPEN_ITER` | Number of opening iterations. Opening mainly removes small isolated noise. |
| `MORPH_CLOSE_ITER` | Number of closing iterations. Closing mainly fills small holes and connects nearby regions. |
| `ENABLE_AREA_FILTER` | Enables contour area filtering. |
| `MIN_REGION_AREA` | Remove detected regions smaller than this area. |
| `MAX_REGION_AREA` | Remove detected regions larger than this area. |

## Logging Configuration: `LOG_CFG`

| Name | Meaning |
| --- | --- |
| `ENABLE_LOGGING` | Enables the logger. |
| `LOG_TO_CONSOLE` | Prints logs to the console. |
| `LOG_TO_FILE` | Writes logs to `run.log`. |
| `LOG_LEVEL` | Logging level, such as `INFO` or `DEBUG`. |
| `LOG_EVERY_N_FRAMES` | Frame interval for progress logging. |
| `SAVE_RUN_CONFIG_JSON` | Saves the full run configuration to `run_config.json`. |

## Visualization And Output Configuration: `VIS_CFG`

### Debug Timing

| Name | Meaning |
| --- | --- |
| `DEBUG_INTERVAL` | Save debug outputs every N frames. For example, `10` saves frame 0, 10, 20, and so on. |
| `COMPARE_SAVE_ONLY_DEBUG_FRAMES` | If `True`, comparison images are saved only on debug frames. If `False`, comparison images are saved for every frame. |
| `QUIVER_STEP` | Spacing between arrows in quiver visualizations. |
| `QUIVER_SCALE` | Arrow length scale in quiver visualizations. |
| `ENABLE_GLOBAL_STAB` | Enables global affine stabilization before optical-flow calculation. This can reduce camera-motion effects, but it is disabled by default. |

### Video Outputs

| Name | Meaning |
| --- | --- |
| `SAVE_VIDEO_HSV` | Saves HSV optical-flow visualization video. Color represents direction and brightness represents speed. |
| `SAVE_VIDEO_QUIVER` | Saves optical-flow arrow video. |
| `SAVE_VIDEO_REGION` | Saves video with detected target contours. |
| `SAVE_VIDEO_BOX` | Saves video with bounding boxes and center points. |
| `SAVE_VIDEO_COMBINED` | Saves video with contours, boxes, center points, and size text together. |

### Debug Outputs

| Name | Meaning |
| --- | --- |
| `ENABLE_DEBUG_OUTPUTS` | Enables algorithm-debug outputs. These files explain why points were classified as target or background-like. |
| `SAVE_DEBUG_PLOTS` | Saves direction/magnitude diagnostic plots. |
| `SAVE_DEBUG_TABLE_CSV` | Saves a per-sample CSV table for debug frames. |
| `SAVE_DEBUG_EXCEL_PRE` | Saves Excel debug workbooks for debug frames. |
| `SAVE_DEBUG_DETECT_FRAME` | Saves final detection images for debug frames. |
| `SAVE_DEBUG_QUIVER_RAW` | Saves raw, unfiltered quiver images for debug frames. |

### Compare Outputs

| Name | Meaning |
| --- | --- |
| `ENABLE_COMPARE_OUTPUTS` | Saves before/after processing comparison images. |
| `SAVE_COMPARE_PRE_MASK_IMAGE` | Saves the target mask before morphology and area filtering. |
| `SAVE_COMPARE_PRE_OVERLAY_IMAGE` | Saves the original frame with the pre-processing mask overlaid. |
| `SAVE_COMPARE_POST_MORPH_MASK_IMAGE` | Saves the mask after morphology. |
| `SAVE_COMPARE_POST_FINAL_MASK_IMAGE` | Saves the final mask after morphology and area filtering. |
| `SAVE_COMPARE_POST_OVERLAY_IMAGE` | Saves the original frame with the final mask overlaid. |
| `SAVE_COMPARE_REGION_IMAGE` | Saves the frame with final contours. |
| `SAVE_COMPARE_BOX_IMAGE` | Saves the frame with final bounding boxes. |
| `SAVE_COMPARE_COMBINED_IMAGE` | Saves the frame with contours and boxes together. |
| `SAVE_COMPARE_EXCEL` | Also saves Excel debug workbooks when comparison output needs sample-level data. |

### Excel Switches

These names describe planned or detailed Excel content controls:

| Name | Meaning |
| --- | --- |
| `EXCEL_SAVE_TEXT_GRID` | Intended switch for text-style grid output. |
| `EXCEL_SAVE_MAG_GRID` | Intended switch for magnitude grid output. |
| `EXCEL_SAVE_ANGLE_GRID` | Intended switch for angle grid output. |
| `EXCEL_SAVE_DTHETA_GRID` | Intended switch for direction-deviation grid output. |
| `EXCEL_SAVE_DMAG_GRID` | Intended switch for magnitude-deviation grid output. |
| `EXCEL_SAVE_TARGET_GRID` | Intended switch for target-classification grid output. |
| `EXCEL_SAVE_INSIDE_DIR_GRID` | Intended switch for inside-dominant-direction grid output. |
| `EXCEL_ENABLE_COLOR` | Intended switch for Excel coloring. |
| `EXCEL_HIGHLIGHT_TARGET` | Intended switch for highlighting target points. |
| `EXCEL_HIGHLIGHT_INSIDE_DIR` | Intended switch for highlighting points inside the dominant direction. |
| `EXCEL_TEXT_SHOW_ANGLE_MAG` | Intended switch for showing angle and magnitude together in text cells. |

Current implementation note: the code currently writes two Excel sheets, `summary` and `sample_points`. The grid-specific switches above are present in configuration, but the current Excel writer does not yet create separate grid sheets for each one.

### Statistics Outputs

| Name | Meaning |
| --- | --- |
| `SAVE_STATS_CSV` | Saves frame-level statistics to `flow_stats_two_stage.csv`. |
| `SAVE_RAW_SPEED_HIST_CSV` | Saves raw optical-flow speed histogram data to CSV. |
| `SAVE_RAW_SPEED_HIST_PNG` | Saves raw optical-flow speed histogram as a PNG chart. |
