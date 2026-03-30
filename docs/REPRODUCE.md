# Reproducibility Guide

This document explains how to reproduce training and evaluation for V2X-MoE.

## 1. Environment
Install dependencies using either:
```bash
conda env create -f environment.yml
conda activate v2x-qa
```
or
```bash
pip install -r requirements.txt
```

## 2. Prepare released annotation files
Place the released V2X-QA JSONL files under:
```text
data/train/
data/test/
```

## 3. Prepare external raw images
Download the original V2X-Seq-SPD raw images from the official source.

Place them under:
```text
data/raw_external/V2X-Seq-SPD-vehicle-side-image/
data/raw_external/V2X-Seq-SPD-infrastructure-side-image/
```

The repository does not redistribute these raw images.

## 4. Run stage 1 training
```bash
python model/train/v2x_moe_train_mcqa_qwen3_stage1.py
```
Default output:
```text
outputs/stage1/
```
Expected saved checkpoints:
```text
outputs/stage1/epoch_1/
outputs/stage1/epoch_2/
outputs/stage1/epoch_3/
outputs/stage1/epoch_4/
```

## 5. Run stage 2 CO refinement
```bash
python model/train/v2x_moe_train_mcqa_qwen3_co_boost.py
```
Default input checkpoint:
```text
outputs/stage1/epoch_4/
```
Default output:
```text
outputs/stage2_co/
```

## 6. Run stage 3 IS refinement
```bash
python model/train/v2x_moe_train_mcqa_qwen3_is_boost.py
```
Default input checkpoint:
```text
outputs/stage2_co/epoch_2/
```
Default output:
```text
outputs/stage3_is/
```

## 7. Evaluate with the released final checkpoint
If you want to evaluate the released public model directly, place the released final adapters under `checkpoints/` and run:
```bash
python model/eval/v2x_moe_eval_mcqa_qwen3.py
```
Default output:
```text
outputs/eval/
```

## 8. What the evaluation script writes
By default, the evaluation script writes:
- reliability plot PNG to `outputs/eval/`

Depending on the script configuration, it can also optionally save:
- per-sample prediction JSONL
- evaluation summary JSON

## 9. Troubleshooting
### Error: raw image directory not found
This usually means the original V2X-Seq-SPD raw images have not been downloaded or placed under the expected local directory structure.

### Error: adapter directory not found
Check that the checkpoint directory contains:
- `vs_expert/`
- `is_expert/`
- `co_expert/`

### Error: wrong expert name
Use `vs_expert`, not `va_expert`.
