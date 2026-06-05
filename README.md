# Farneback Optical Flow Motion Detector

This project detects moving target regions in video using OpenCV Farneback optical flow. It creates visual debug videos, masks, region overlays, CSV statistics, and optional Excel debug grids.

## Features

- Two-stage direction and magnitude filtering for motion detection
- Optional forward-backward optical-flow reliability diagnosis
- Optional color, edge, and texture appearance-change detection
- Configurable fusion of optical-flow and appearance-change masks
- Optional recent-frame temporal mask filtering
- Morphology and area filtering for cleaner target regions
- HSV flow visualization, quiver arrows, region overlays, and bounding boxes
- Per-run output folders with logs, configuration snapshots, CSV stats, and debug images
- Optional Excel debug sheets for frame-level inspection

## Project Structure

```text
.
├── main_flow_detect.py
├── main_flow_detect_temporal.py
├── VERSION_NOTES.md
├── CHANGELOG.md
├── tools/
│   ├── flow_visual_debug.py
│   ├── flow_reliability.py
│   ├── appearance_change.py
│   ├── temporal_filter.py
│   └── save_manager.py
├── input_video/
│   └── .gitkeep
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Put a video file in `input_video/`, then run:

```bash
python main_flow_detect.py input_video/cam2.mp4
```

You can also choose a custom output folder:

```bash
python main_flow_detect.py input_video/cam2.mp4 --output-dir flow_outputs
```

For faster runs without Excel files:

```bash
python main_flow_detect.py input_video/cam2.mp4 --no-excel
```

The current stable version is Version 3.1. `main_flow_detect.py` provides the
appearance-adjusted detector, and `main_flow_detect_temporal.py` adds temporal
filtering.

The current appearance configuration starts with color change only:

```python
APPEARANCE_CFG = dict(
    ENABLE_APPEARANCE_CHANGE=True,
    FUSION_MODE="flow_or_appearance",
    COLOR_DIFF_THRESHOLD=35.0,
    ENABLE_EDGE_CHANGE=False,
    ENABLE_TEXTURE_CHANGE=False,
    ...
)
```

Run the current temporal entry point with:

```bash
python main_flow_detect_temporal.py input_video/cam2.mp4
```

The temporal program calls `tools/temporal_filter.py` after separately
postprocessed flow/appearance fusion and before final morphology and area
filtering. Its current rule keeps pixels that appear in at least `2` of the
recent `5` frames.

See [VERSION_NOTES.md](VERSION_NOTES.md) for current stable-version details and
[CHANGELOG.md](CHANGELOG.md) for the version summary. Historical complete
versions remain available through Git tags.

## Outputs

Each run creates a numbered folder under the output directory. Typical outputs include:

- `videos/`: HSV flow, quiver, region, box, and combined videos
- `stats/`: frame-level motion statistics and raw speed histogram
- `cluster_plots/`: debug plots for direction and magnitude thresholds
- `compare_pre/` and `compare_post/`: mask and overlay comparison images
- `excel/`: optional grid-level debug workbooks
- `run.log` and `run_config.json`: execution log and saved parameters
- `appearance_debug/`: optional appearance masks, difference images, and fused candidate masks
- `flow_reliability/`: optional forward-backward consistency diagnostic images
- `temporal_debug/`: temporal input, output, hit-count, and confidence images

## Notes

Video files and generated outputs are ignored by Git by default. Keep large local videos outside the repository, or place only small demo files in `input_video/` when you intentionally want to share them.
