# Current Stable Version Notes: 3.1

Version 3.1 keeps the Version 3.0 temporal filter and adds the Version 2.1
appearance morphology adjustments.

Historical source trees and version-specific notes remain available through the
Git tags `v2.0`, `v2.1`, `v3.0`, and `v3.1`.

## Processing order

```text
raw optical-flow mask -> separate flow postprocessing
raw appearance cues   -> separate appearance postprocessing
                     -> flow/appearance fusion
                     -> Version 3.0 temporal filtering
                     -> final morphology and area filtering
```

## Version relationship

- Version 2.1: adjusted appearance morphology and fusion without temporal
  filtering.
- Version 3.0: original temporal implementation before appearance adjustment.
- Version 3.1: Version 2.1 appearance adjustment plus Version 3.0 temporal
  filtering.

Version 3.1 writes to the separate output prefix
`flow_outputs_temporal_v3_1`.

The current tuned temporal setting uses:

```python
HISTORY_LENGTH=5
MIN_HIT_FRAMES=2
MIN_HISTORY_FRAMES=3
```

This is less strict than the initial Version 3.1 experiment, which required
three hits and removed too many short-lived detections.
