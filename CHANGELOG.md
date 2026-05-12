# Changelog

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
