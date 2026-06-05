# Changelog / 更新日志

The `main` branch contains only the current stable source and
`VERSION_NOTES.md`. Complete historical source trees remain available through
the corresponding Git tags. Future changelog entries must be written in both
English and Chinese.

`main` 分支只保留当前稳定版源码和 `VERSION_NOTES.md`。各历史版本的完整源码通过对应
Git 标签保存。今后的更新日志必须同时使用英文和中文书写。

## Version 3.1

### English

- Kept the Version 3.0 recent-frame temporal filter.
- Integrated the Version 2.1 separately postprocessed appearance and flow masks.
- Changed the temporal output prefix to `flow_outputs_temporal_v3_1`.
- Tuned the temporal rule from three required hits to two required hits within
  the recent five frames, reducing lost detections.

### 中文

- 保留 Version 3.0 的近期帧时序过滤功能。
- 集成 Version 2.1 中分别后处理的外观掩膜和光流掩膜。
- 将时序输出目录前缀改为 `flow_outputs_temporal_v3_1`。
- 将最近五帧所需命中次数从三次调整为两次，减少目标被时序过滤完全删除的情况。

## Version 3.0

### English

Version 3.0 starts the temporal-processing development line without replacing
the Version 2.0 main program.

- Added the separate `main_flow_detect_temporal.py` entry point.
- Added the standalone `tools/temporal_filter.py` tool.
- Added recent-frame mask voting with configurable history length and minimum
  hit count.
- Added optional historical-mask dilation to tolerate small movement and mask
  jitter.
- Added configurable warmup behavior and an option to retain only current-mask
  pixels.
- Added temporal debug images, CSV/log statistics, and unit tests.

### 中文

Version 3.0 开始时序处理开发线，同时保留 Version 2.0 主程序。

- 新增独立入口程序 `main_flow_detect_temporal.py`。
- 新增独立时序工具 `tools/temporal_filter.py`。
- 新增近期帧掩膜投票，可配置历史长度和最小命中次数。
- 支持对历史掩膜进行膨胀，以容忍轻微目标移动和掩膜抖动。
- 新增可配置预热行为，以及只保留当前帧候选像素的选项。
- 新增时序调试图、CSV/日志统计和单元测试。

## Version 2.1

### English

- Postprocesses optical-flow and appearance masks separately before fusion.
- Starts appearance processing with color change only.
- Tightens appearance thresholds and adds appearance-region area filtering.
- Keeps final postprocessing after fusion.

### 中文

- 光流掩膜和外观掩膜分别后处理后再融合。
- 外观检测默认只启用颜色变化。
- 收紧外观阈值，并增加外观区域面积过滤。
- 融合完成后仍保留最终后处理步骤。

## Version 2.0

### English

Version 2.0 adds appearance-change detection and mask fusion while retaining the
forward-backward optical-flow reliability diagnostics introduced during Version
1.2 development.

- Added `tools/appearance_change.py`.
- Added HSV color-change, Canny edge-change, and Laplacian texture-change
  detection.
- Added morphology cleanup and configurable optical-flow/appearance fusion.
- Added optional `appearance_debug/` output images.
- Added `appearance_pixels` and `fused_target_pixels` to CSV statistics and logs.
- Added the Version 1.2 reliability tool and statistics.

### 中文

Version 2.0 新增外观变化检测与掩膜融合，同时保留 Version 1.2 开发阶段加入的
前后向光流可靠性诊断。

- 新增 `tools/appearance_change.py`。
- 新增 HSV 颜色变化、Canny 边缘变化和 Laplacian 纹理变化检测。
- 新增形态学清理及可配置的光流/外观掩膜融合。
- 新增可选的 `appearance_debug/` 调试图片。
- CSV 和日志新增 `appearance_pixels` 与 `fused_target_pixels`。
- 加入 Version 1.2 的可靠性工具和统计字段。

## Version 1.2

### English

- Added reusable forward-backward optical-flow reliability diagnostics.
- Added reliability masks, error images, and summary statistics.
- Kept reliability diagnosis as diagnostic-only by default.

### 中文

- 新增可复用的前后向光流可靠性诊断。
- 新增可靠性掩膜、误差图和汇总统计。
- 默认只将可靠性检查用于诊断，不直接过滤检测结果。

## Version 1.1

### English

- Added `USE_ALL_POINTS_FOR_MAG0` to calculate the magnitude baseline from all
  sampled pixels.
- Added `mag0_source` to logs and CSV statistics.
- Retained the original direction-filtered baseline option.

### 中文

- 新增 `USE_ALL_POINTS_FOR_MAG0`，可使用全部采样点计算速度基准。
- 日志和 CSV 统计新增 `mag0_source`。
- 保留原有的方向筛选点速度基准模式。

## Version 1.0

### English

Initial uploaded version of the Farneback optical-flow motion detector.

### 中文

Farneback 光流运动目标检测器的初始上传版本。
