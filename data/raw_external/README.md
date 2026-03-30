# External Raw Data Placeholder

This directory is intentionally included as a placeholder only.

## Important
The original V2X-Seq-SPD raw images are **not distributed in this repository**.
You must download them separately from the official source and place them locally here.

## Expected directory layout
```text
data/raw_external/
├── V2X-Seq-SPD-vehicle-side-image/
└── V2X-Seq-SPD-infrastructure-side-image/
```

## Purpose
The released V2X-QA JSONL files store image identifier fields such as:
- `image_id_vs`
- `image_id_is`

The training and evaluation scripts use these IDs to locate the raw image files in the directories above.

## Reminder
Please comply with the original license and usage terms of the upstream dataset source when downloading and using the raw images.
