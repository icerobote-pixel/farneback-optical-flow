# Version 3.0 Temporal Processing Notes

## Version strategy

Version 3.0 begins the temporal-processing development line. It does not modify
or replace the Version 2.0 entry point:

```text
main_flow_detect.py
```

The temporal version has its own entry point:

```text
main_flow_detect_temporal.py
```

Both programs can therefore be run independently for direct comparison.

## Standalone temporal tool

The new temporal logic lives in:

```text
tools/temporal_filter.py
```

`TemporalMaskFilter` stores recent candidate masks and filters short-lived
noise using recent-frame voting.

The Version 3.0 default rule is:

```python
HISTORY_LENGTH=5
MIN_HIT_FRAMES=3
MIN_HISTORY_FRAMES=3
MOTION_TOLERANCE_DILATE_ITER=1
KEEP_CURRENT_ONLY=True
WARMUP_MODE="passthrough"
```

After the first three frames, a current candidate pixel is retained only when
its location, or a nearby location allowed by dilation, has appeared in at
least three recent masks.

## Processing order

```text
optical-flow candidate mask
        |
optional appearance-change fusion
        |
Version 3.0 temporal mask filter
        |
morphology and area filtering
        |
final detected regions
```

## Debug outputs

When enabled, `temporal_debug/` contains:

- `temporal_input_mask_frame_XXXXX.png`
- `temporal_output_mask_frame_XXXXX.png`
- `temporal_hit_count_frame_XXXXX.png`
- `temporal_confidence_frame_XXXXX.png`

## Run

```bash
python main_flow_detect_temporal.py input_video/cam2.mp4
```

Version 3.0 uses the separate output prefix `flow_outputs_temporal_v3`.
