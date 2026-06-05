# Version 2.1 Appearance Morphology Notes

Version 2.1 adjusts the appearance-change experiment after Version 2.0 showed
that raw `flow_or_appearance` fusion was too permissive.

## Main changes

- Optical-flow and appearance masks are postprocessed separately before fusion.
- Appearance processing starts with color change only.
- Edge and texture change are disabled by default because they were too
  sensitive for the tested scene.
- Color threshold increased from `22` to `35`.
- Texture threshold increased from `18` to `30` for later experiments.
- Edge dilation is disabled.
- Appearance opening increased from `0` to `1`.
- Appearance regions smaller than `300` pixels or larger than `50000` pixels
  are removed before fusion.

The final fused mask still receives the normal final morphology and area
filtering.
