# DistriNVS framework figure

This directory contains the editable source and export formats for Figure 2.

- `make_framework.py`: Python source for both the paper figure and PPTX.
- `distrinvs_framework.svg`: primary editable vector export.
- `distrinvs_framework.pdf`: vector asset included by LaTeX.
- `distrinvs_framework.png`: 600 dpi preview/export.
- `distrinvs_framework.tiff`: 600 dpi LZW-compressed export.
- `distrinvs_framework_editable.pptx`: three-slide deck. Slide 1 is rebuilt
  entirely from native PowerPoint shapes; slide 2 is a traceable
  `no_hard_composition` single-frame walkthrough; slide 3 is an editing guide
  and palette.
- `extract_nohard_intermediates.py`: reruns one held-out frame through the
  step-10,000 no-hard checkpoint and exports additional model internals.
- `intermediate_assets/`: display PNGs, raw arrays, and provenance for the
  walkthrough slide.

Regenerate in the project directory with:

```bash
conda run --no-capture-output -n latex python figures/distrinvs_framework/make_framework.py
```

To refresh the real-case walkthrough without touching the paper PDF/SVG/PNG:

```bash
conda run --no-capture-output -n foundation_stereo \
  python figures/distrinvs_framework/extract_nohard_intermediates.py --device cuda:1
conda run --no-capture-output -n latex \
  python figures/distrinvs_framework/make_framework.py --pptx-only
```

The archived warp and all intermediate predictions on slide 2 form one exact
held-out chain for `session_004_scene_2_tool_3/frame_000002`. The separately
requested `tool_4` RGB belongs to the checkpoint's training split, so it is
shown only as a dashed-border appearance example and is not represented as the
source of the `tool_3` warp.

The intended paper size is 183 × 76 mm. Solid arrows are inference-time paths;
purple dashed arrows and the lower lane are training only.
