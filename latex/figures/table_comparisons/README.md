# Table I and Table II qualitative comparisons

These image plates follow the qualitative-grid convention in the comparison
report. Table I retains its original fixed samples; Table II uses a declared
PSNR-advantage selection rule.

## Table I

`table1_qualitative_comparison` contains two matched blocks:

- Gaussian reconstruction: source RGB, seven Gaussian baselines, DistriNVS,
  and target RGB.
- Image completion: DSS warp, LDM, LaMa, LCM-LoRA, Hunyuan-DiT, DistriNVS,
  and target RGB.

The rows are the 100th retained sample from the lexicographically first and
last iMED validation scenes:

- `session_004_scene_2_tool_3/frame_000299`
- `session_007_scene_5_tool_2/frame_000299`

## Table II

`table2_zeroshot_qualitative_comparison` compares the same five methods as
Table II alongside the common source RGB, DSS warp, and target. For each dataset, the rows are
the two positive-margin samples with the largest DistriNVS PSNR advantage over
the strongest displayed comparator, constrained to different keyframe
sequences and raw-hole ratio at least 10%:

- `endovis_dataset_8_keyframe_3/frame_000005`
- `endovis_dataset_8_keyframe_1/frame_000013`
- `endovis_dataset_9_keyframe_0/frame_000067`
- `endovis_dataset_9_keyframe_2/frame_000003`

The teal outline identifies DistriNVS; it does not indicate metric rank.
White labels report the corresponding per-frame PSNR without a backing band;
source, DSS-warp, and target panels are not assigned a PSNR label.

## Reproduction and assets

Run:

```bash
conda run --no-capture-output -n latex python make_table_visualizations.py --output-dir .
```

Each figure is exported as editable PDF/SVG and 600-dpi PNG/TIFF. Text uses
Times New Roman. Images are displayed as RGB, resized to target dimensions
only when necessary, and never cropped or enhanced per image. Table-II source
RGB uses the same bilinear resize as the model input pipeline. The two
`*_sources.csv` files record every panel's source path and display transform;
`qualitative_provenance.json` records software, font, color semantics, and
sample-selection rules.
