# Changelog / 更新日志

## Version 1.2 / 版本 1.2

### English

Version 1.2 adds optical-flow reliability diagnostics based on forward-backward consistency.

Changes included:

- Added `tools/flow_reliability.py` as a reusable and standalone reliability diagnosis tool.
- Added `RELIABILITY_CFG` to control forward-backward consistency checks.
- Added `flow_reliability/` output images: `fb_error_frame_XXXXX.png` and `reliable_flow_mask_frame_XXXXX.png`.
- Added `mean_fb_error` and `reliable_pixel_ratio` to frame-level CSV statistics and logs.
- Kept `APPLY_RELIABLE_MASK_TO_DETECTION=False` by default so reliability is diagnostic only and does not suppress detected targets.
- Reduced heavy debug table/Excel output defaults while keeping comparison images and reliability diagnostics available.

### 中文

版本 1.2 新增了基于前后向一致性的光流可靠性诊断功能。

主要变化：

- 新增 `tools/flow_reliability.py`，可作为主程序调用的工具，也可以单独运行进行可靠性诊断。
- 新增 `RELIABILITY_CFG`，用于控制前后向一致性检测。
- 新增 `flow_reliability/` 输出图像：`fb_error_frame_XXXXX.png` 和 `reliable_flow_mask_frame_XXXXX.png`。
- 在逐帧 CSV 统计和日志中新增 `mean_fb_error` 与 `reliable_pixel_ratio`。
- 默认保持 `APPLY_RELIABLE_MASK_TO_DETECTION=False`，也就是可靠性检测只做诊断，不直接过滤目标，避免加重漏检。
- 默认减少较重的调试表格和 Excel 输出，同时保留对比图和可靠性诊断输出。

## Version 1.1 / 版本 1.1

### English

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

### 中文

版本 1.1 更新了两阶段光流检测中的速度基准 `mag0` 计算方式。

在版本 1.0 中，`mag0` 由通过主方向筛选的点计算得到。版本 1.1 增加了 `USE_ALL_POINTS_FOR_MAG0=True` 选项，可以使用全部采样点计算速度基准。这让速度阈值拥有更全局的参考，便于在主方向筛选不稳定或过于严格时进行对比实验。

主要变化：

- 在 `ALG_CFG` 中新增 `USE_ALL_POINTS_FOR_MAG0`。
- 当前启用的速度基准模式从 `inside_direction` 中位数改为 `all_points` 中位数。
- 新增 `mag0_source`，用于记录 `mag0` 来自全部采样点、方向筛选点，还是 fallback 逻辑。
- 在日志和 CSV 统计中加入 `mag0_source`，便于实验对比。
- 根据新的实验配置调整了方向和速度阈值缩放参数。
- 当 `USE_ALL_POINTS_FOR_MAG0=False` 时，仍保留原来的方向筛选点中位数逻辑。
- 本版本没有修改 `tools` 模块。

## Version 1.0 / 版本 1.0

### English

Initial uploaded version of the Farneback optical flow motion detector. The magnitude baseline `mag0` was calculated from points that passed the dominant-direction filter.

### 中文

首次上传的 Farneback 光流运动目标检测版本。该版本中，速度基准 `mag0` 由通过主方向筛选的点计算得到。
