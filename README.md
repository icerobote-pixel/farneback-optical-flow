# Farneback Optical Flow Motion Detector

This project detects moving target regions in video using OpenCV Farneback optical flow. It creates visual debug videos, masks, region overlays, CSV statistics, and optional Excel debug workbooks.

## Features

- Two-stage direction and magnitude filtering for motion detection
- Morphology and area filtering for cleaner target regions
- HSV flow visualization, quiver arrows, region overlays, and bounding boxes
- Per-run output folders with logs, configuration snapshots, CSV stats, and debug images
- Optional Excel debug sheets for frame-level inspection

## Project Structure

```text
.
|-- main_flow_detect.py
|-- tools/
|   |-- flow_visual_debug.py
|   `-- save_manager.py
|-- input_video/
|   `-- .gitkeep
|-- docs/
|   |-- config_reference.md
|   `-- output_explanation.md
|-- requirements.txt
`-- README.md
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

The current script accepts the first command-line argument as the input video path. Other options are configured inside `main_flow_detect.py` through `ALG_CFG`, `POST_CFG`, `LOG_CFG`, and `VIS_CFG`.

## Documentation

- [Configuration reference](docs/config_reference.md): explains the optional switches and tuning parameters in the main program.
- [Output explanation](docs/output_explanation.md): explains each output folder, video, image, CSV file, and Excel sheet.

## Outputs

Each run creates a numbered folder under `OUTPUT_BASE_ROOT`. Typical outputs include:

- `videos/`: HSV flow, quiver, region, box, and combined videos
- `stats/`: frame-level motion statistics and raw speed histogram files
- `cluster_plots/`: debug plots for direction and magnitude thresholds
- `compare_pre/` and `compare_post/`: pre/post-processing mask and overlay comparison images
- `excel/`: optional frame-level debug workbooks
- `run.log` and `run_config.json`: execution log and saved parameters

## Notes

Video files and generated outputs are ignored by Git by default. Keep large local videos outside the repository, or place only small demo files in `input_video/` when you intentionally want to share them.
