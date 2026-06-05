# Version 2.0 Backup Notes

## Version purpose

Version 2.0 backs up the latest local detector from:

```text
E:\Project\Optical Flow\Farneback_test\modified_main_flow_detect_appearance.py
```

The GitHub version uses the repository-standard filename:

```text
main_flow_detect.py
```

## Main addition

The detector now calls `tools/appearance_change.py` to optionally identify
changes that optical flow alone may miss.

The tool contains:

- HSV color-change detection
- Canny edge-change detection
- Laplacian texture-change detection
- Appearance-mask morphology cleanup
- Optical-flow and appearance-mask fusion
- Appearance debug-image saving

## Default behavior

The latest backed-up runtime configuration keeps the new feature disabled:

```python
ENABLE_APPEARANCE_CHANGE=False
```

This preserves the previous optical-flow-only detection behavior. To test the
new tool, set it to `True`. The configured fusion mode is:

```python
FUSION_MODE="flow_or_appearance"
```

With that mode enabled, a pixel is retained when either optical flow or the
appearance detector marks it as a candidate.

## Additional included development

This version also includes the Version 1.2 development work:

- Forward-backward optical-flow reliability diagnosis
- `mean_fb_error` and `reliable_pixel_ratio` statistics
- Optional reliability debug images

## Restore and run

Install dependencies and run:

```bash
pip install -r requirements.txt
python main_flow_detect.py input_video/cam2.mp4
```

Video inputs and generated output directories are intentionally excluded from
the GitHub backup.

## Local result backup

The local Version 2.0 backup also includes two complete comparison runs:

- `flow_outputs_two_stage_dir_mag_compare_run_014`: appearance-change detection
  enabled with `flow_or_appearance` fusion.
- `flow_outputs_two_stage_dir_mag_compare_run_015`: appearance-change detection
  disabled, representing the latest default runtime configuration.

These generated outputs are kept only in the local backup because they are too
large for the source-code GitHub repository.
