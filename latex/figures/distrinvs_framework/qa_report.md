# PPTX QA report

## Deliverable

- File: `distrinvs_framework_editable.pptx`
- Canvas: 13.333 × 5.555 in (the existing 183 × 76 mm aspect ratio)
- Slides: 3
  1. Fully editable conceptual framework
  2. Traceable no-hard-composition single-frame walkthrough
  3. Editing guide and palette
- Embedded evidence images: 12

## Scientific verification

- Checkpoint: `no_hard_composition/seed_6666/step_0010000.pt`
- Checkpoint step: 10,000
- Matched sample: `session_004_scene_2_tool_3/frame_000002`
- `hard_composition`: false
- Post-processing: disabled
- Recomputed DSS warp versus archived PNG: maximum uint8 error = 0
- Reconstructed `(1-g) transport + g synthesis` versus model output: maximum absolute error = 0
- The separately requested tool_4 RGB belongs to the checkpoint training split and is explicitly marked as a dashed-border training example, not as the source of the tool_3 warp.

## Self-review and corrections

- High-severity defects: none.
- Medium-severity defect corrected: the training example was moved below the matched source so the primary reading path cannot imply that tool_4 generated the tool_3 warp.
- Medium-severity defect corrected: an opaque lane-colored label backing prevents the training-only dashed arrow from crossing the `predicted RGB-D + risk` text on slide 1.
- Images use their native 5:4 aspect ratio; no warp, mask, or invalid black region is cropped.

## Final verification

- PPTX package ZIP test: passed.
- Reopen with `python-pptx`: passed.
- Out-of-slide shapes: 0.
- Placeholder tokens: 0.
- LibreOffice headless rendered all 3 slides successfully; the rendered pages were visually inspected for missing images, overlap, clipping, and unreadable labels.
- Speaker notes were not added because this file is an editable manuscript-figure deck rather than a talk deck.

## Known display convention

Heatmaps are colorized for visualization. Fixed-range maps use [0,1]; unbounded
statistics use the display ranges recorded in `intermediate_assets/manifest.json`.
The untouched float tensors remain available in `intermediates_raw.npz`.
