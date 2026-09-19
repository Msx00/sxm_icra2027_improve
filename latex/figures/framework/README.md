# DSS output visualization

This directory contains a checkpoint-derived visualization of the outputs of
distributional surface splatting (DSS). The selected held-out iMED sample is
`session_004_scene_2_tool_3/frame_000428`; its large unsupported region makes
the transport boundary and reliability statistics visible.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/data/mashixing/dataset_18tb/icra-2027 \
/home/data/mashixing/miniconda3/envs/foundation_stereo/bin/python3.11 \
visualize_dss_outputs.py --device cuda:0
```

The script loads the `no_hard_composition` checkpoint at step 10,000 and
exports the source RGB, warped RGB, 16-channel warped features, expected depth,
support, coverage, depth variance, collision entropy, geometry confidence, and
valid mask. The feature tensor is displayed using a deterministic three-axis
PCA projection. Raw arrays are retained in `dss_outputs_raw.npz`.

## Display policy

- RGB images are resized by the dataset loader only; they are not cropped or
  photometrically enhanced.
- Scalar maps are pseudo-colored using the ranges recorded in
  `dss_outputs_manifest.json`.
- Expected depth and variance use the 1st--99th percentile range over valid DSS
  pixels. Support uses zero to its valid-pixel 99th percentile.
- Coverage, collision entropy, and confidence use the fixed interval [0, 1].
- Invalid pixels are dark gray in depth, variance, and collision maps.

The assembled figure is provided as editable PDF/SVG and as 600-dpi PNG/TIFF.
