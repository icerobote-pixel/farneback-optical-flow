# Changelog

The `main` branch contains only the current stable source and
`VERSION_NOTES.md`. Complete historical source trees remain available through
the corresponding Git tags.

## Version 3.1

- Kept the Version 3.0 recent-frame temporal filter.
- Integrated the Version 2.1 separately postprocessed appearance and flow masks.
- Changed the temporal output prefix to `flow_outputs_temporal_v3_1`.

## Version 3.0

Version 3.0 starts the temporal-processing development line without replacing
the Version 2.0 main program.

Changes included:

- Added the separate `main_flow_detect_temporal.py` entry point.
- Added the standalone `tools/temporal_filter.py` tool.
- Added recent-frame mask voting with configurable history length and minimum
  hit count.
- Added optional dilation of historical masks to tolerate small target
  movement or mask jitter.
- Added configurable warmup behavior and an option to retain only pixels in
  the current candidate mask.
- Added `temporal_debug/` images for temporal input, output, hit count, and
  confidence.
- Added temporal pixel count, history length, and readiness to CSV statistics
  and logs.
- Added unit tests for repeated detections, single-frame noise, and positional
  tolerance.

## Version 2.1

- Postprocesses optical-flow and appearance masks separately before fusion.
- Starts appearance processing with color change only.
- Tightens appearance thresholds and adds appearance-region area filtering.
- Keeps final postprocessing after fusion.

## Version 2.0

Version 2.0 adds appearance-change detection and mask fusion while retaining the
forward-backward optical-flow reliability diagnostics introduced during Version
1.2 development.

Changes included:

- Added `tools/appearance_change.py`.
- Added color-change detection in HSV space.
- Added edge-change detection using Canny edges.
- Added texture-change detection using Laplacian responses.
- Added morphology cleanup for the combined appearance mask.
- Added configurable fusion between optical-flow and appearance masks:
  - `flow_only`
  - `appearance_only`
  - `flow_and_appearance`
  - `flow_or_appearance`
- Added optional `appearance_debug/` output images.
- Added `appearance_pixels` and `fused_target_pixels` to CSV statistics and logs.
- Added `tools/flow_reliability.py` and reliability statistics from the Version
  1.2 development state.
- Kept appearance-change detection disabled in the backed-up default
  configuration so existing optical-flow behavior remains the default.

## Version 1.1

Version 1.1 updates the magnitude filtering baseline used in the two-stage optical flow detection process.

In version 1.0, the speed baseline `mag0` was calculated from pixels that passed the dominant-direction filter. In version 1.1, `mag0` can now be calculated from all sampled pixels by enabling `USE_ALL_POINTS_FOR_MAG0=True`. This gives the magnitude threshold a more global reference and makes it easier to compare results when the dominant-direction filtering stage is unstable or too restrictive.

Changes included:

- Added `USE_ALL_POINTS_FOR_MAG0` to `ALG_CFG`.
- Changed the enabled magnitude baseline mode from `inside_direction` median to `all_points` median.
- Added `mag0_source` to record whether `mag0` came from all sampled points, direction-filtered points, or fallback logic.
- Added `mag0_source` to logs and CSV statistics for easier experiment comparison.
- Adjusted direction and magnitude threshold scaling parameters for the new experiment configuration.
- Kept the original direction-filtered median logic available when `USE_ALL_POINTS_FOR_MAG0=False`.
- No changes were made to the `tools` modules.

## Version 1.0

Initial uploaded version of the Farneback optical flow motion detector. The magnitude baseline `mag0` was calculated from points that passed the dominant-direction filter.
