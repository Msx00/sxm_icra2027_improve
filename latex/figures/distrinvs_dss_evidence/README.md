# DSS evidence and soft-fusion routing figure

This figure supports two linked claims: continuous DSS support is thresholded
into the render-validity mask, and the full reliability condition produces a
continuous expert-fusion gate rather than a binary inpainting mask.

The representative sample is the held-out iMED validation frame
`session_004_scene_2_tool_3/frame_000002`. Its raw source-view overlap is
42.31%, leaving a 57.69% raw hole that makes the transport-to-synthesis routing
visually explicit. `dss_evidence_data.npz` stores the same-frame DSS warp,
unnormalized support, exact completed render-validity mask, and effective
synthesis gate from the primary no-hard/soft-fusion checkpoint.
`extract_dss_fusion_data.py` verifies pixelwise that the cached mask equals
`support >= 0.03`, that hard composition is disabled, and that the model output
satisfies the soft-fusion identity before writing the cache.

The support colors are capped at the framewise 99th percentile only for display;
the binary mask is computed from uncapped support. The cyan contour in panel (b)
is the threshold boundary shown in panel (c). Panel (d) uses the fixed gate range
`[0, 1]`, where zero selects the bounded transport expert and one selects the
synthesis expert. No Telea or other RGB post-processing enters the figure.

The four panels use a compact single-row, single-column layout. Their concise
panel titles remain legible at final print size, while the full definitions
are retained in the manuscript caption. The horizontal gate colorbar is tucked
directly below panel (d), keeping the scale visually associated with the
quantity it encodes without adding a fifth vertical strip.

Regenerate the cache with the project inference environment:

```bash
/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11 \
  extract_dss_fusion_data.py --device cuda
```

Build the figure with the manuscript plotting environment:

```bash
conda run --no-capture-output -n sammorph python make_dss_evidence.py
```

All figure text is set in Times New Roman, including math labels. The generator expects
the regular, bold, italic, and bold-italic faces in
`~/.local/share/fonts/msttcorefonts` and stops with an error instead of silently
substituting another serif face. A different installation directory can be
passed with `--font-dir`.

Primary output is editable SVG. PDF is used by LaTeX, and 600-dpi PNG/TIFF files
are supplied for raster-based submission workflows.
