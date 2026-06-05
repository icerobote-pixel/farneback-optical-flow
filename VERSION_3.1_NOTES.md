# Version 3.1 Temporal Processing Notes

Version 3.1 keeps the Version 3.0 temporal filter and adds the Version 2.1
appearance morphology adjustments.

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
