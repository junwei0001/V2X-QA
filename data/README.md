# V2X-QA Data

This directory contains the released annotation files for V2X-QA.

## Contents
- `train/`: released training JSONL annotation files
- `test/`: released testing JSONL annotation files
- `schema.md`: field definitions for all JSONL files
- `raw_external/`: placeholder directory for externally downloaded raw images

## Important note
The original vehicle-side and infrastructure-side images are derived from **V2X-Seq-SPD** and are **not redistributed in this repository**.

To use the released annotations, you must separately download the raw images from the official source and place them under the expected local paths described in `raw_external/README.md`.

## File format
Each split is stored in JSONL format. Each line corresponds to a single QA instance.

## Recommended layout
```text
data/
├── train/
├── test/
├── raw_external/
│   ├── V2X-Seq-SPD-vehicle-side-image/
│   └── V2X-Seq-SPD-infrastructure-side-image/
├── README.md
└── schema.md
```

## Split notes
The public annotation files are intended to support reproducible training and evaluation with the scripts provided in this repository.

If you release answer-bearing test annotations, users can reproduce benchmark results locally.
If you prefer a hidden-label benchmark setting, you may additionally maintain a private test-answer file outside the public repository.

## Compatibility
The training and evaluation scripts in `model/` expect the fields documented in `schema.md` and rely on the raw image IDs to resolve the original vehicle-side and infrastructure-side image files.
