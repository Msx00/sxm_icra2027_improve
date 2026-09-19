# DistriNVS first-page teaser

The recommended asset for LaTeX is `distrinvs_teaser.pdf` at single-column
width. `distrinvs_teaser.svg` keeps all labels as vector text, and
`distrinvs_teaser.png` is the 600-dpi visual-QA preview.

`distrinvs_teaser_editable.pptx` contains:

1. the complete teaser rebuilt from editable PowerPoint shapes and text; and
2. an editing guide with the palette and suggested caption.

Rebuild all outputs with:

```bash
conda run --no-capture-output -n latex python figures/distrinvs_teaser/make_teaser.py
```

Design constraint: source blue and target purple always denote two physically
separate endoscopes. Teal denotes transported/trusted evidence; amber denotes
target-only synthesized content and predicted risk.
