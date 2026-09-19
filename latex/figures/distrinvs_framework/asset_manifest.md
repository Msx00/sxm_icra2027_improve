# Asset manifest — no-hard-composition walkthrough

The walkthrough uses the `no_hard_composition` checkpoint at step 10,000 with
post-processing disabled. The matched inference chain is
`session_004_scene_2_tool_3/frame_000002`.

| Asset | Origin | Scientific role | PPT placement |
|---|---|---|---|
| `00_requested_tool4_train_exemplar.png` | User-selected `session_004_scene_2_tool_4/endoscope2/L/frame_000002.png`, resized to 640×512 | Training-domain appearance example only; not paired with the tool_3 warp | Slide 2, Input Context, dashed frame |
| `01_matched_source_rgb.png` | Model-preprocessed tool_3 `endoscope2/L/frame_000002.png` | Actual RGB input of the matched forward pass | Slide 2, Input Context |
| `02_dss_warp.png` | `prediction["warped_rgb"]`; byte-identical to the archived user-selected warp | DSS transported evidence | Slide 2, DSS Geometry |
| `05_dss_support.png` | `prediction["support"]`, magma display mapping | Target-plane support | Slide 2, DSS Geometry |
| `06_dss_variance.png` | `prediction["render_variance"]`, q99 display mapping | Target-plane depth variance | Slide 2, DSS Geometry |
| `07_collision_entropy.png` | `prediction["collision_entropy"]`, fixed [0,1] display range | Collision ambiguity | Slide 2, DSS Geometry |
| `08_physics_prior.png` | `prediction["physics_prior"]`, fixed [0,1] display range | Analytic synthesis prior | Slide 2, Soft Routing + Experts |
| `09_transport_rgb.png` | `prediction["transport_rgb"]` | Transport-expert RGB estimate | Slide 2, Soft Routing + Experts |
| `10_synthesis_rgb.png` | `prediction["synthesis_rgb"]` | Fourier synthesis-expert RGB estimate | Slide 2, Soft Routing + Experts |
| `11_synthesis_gate.png` | `prediction["synthesis_gate"]`, fixed [0,1] display range | Effective no-hard mixture gate | Slide 2, Soft Routing + Experts |
| `12_soft_fused_rgb.png` | `prediction["target_rgb"]` | Soft-fused output | Slide 2, Output |
| `13_predicted_risk.png` | `prediction["risk"]`, fixed [0,1] display range | Predicted target risk | Slide 2, Output |

Additional exported but undisplayed assets include source depth mean/scale,
render depth, source/render confidence, source normal, completed-valid mask,
trusted mask, and target RGB reference. Raw float arrays are preserved in
`intermediate_assets/intermediates_raw.npz`; exact paths, hashes, display
ranges, and numerical identity checks are recorded in
`intermediate_assets/manifest.json`.
