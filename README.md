# V2X-QA

Official repository for **V2X-QA** and **V2X-MoE**, a multi-view visual question answering dataset, benchmark, and baseline for autonomous driving across **vehicle-side (VS)**, **infrastructure-side (IS)**, and **cooperative (CO)** views.

> Paper: **V2X-QA: A Comprehensive Reasoning Dataset and Benchmark for Multimodal Large Language Models in Autonomous Driving Across Ego, Infrastructure, and Cooperative Views**
>
> arXiv: **REPLACE_WITH_ARXIV_ID**
>
> Paper link: **REPLACE_WITH_ARXIV_LINK**

## News
- Initial public release of the V2X-QA repository.

## Overview
V2X-QA is a real-world multi-view autonomous driving VQA dataset and benchmark built on top of V2X-Seq-SPD. It supports controlled evaluation under three evidence conditions:
- **VS**: vehicle-side reasoning
- **IS**: infrastructure-side reasoning
- **CO**: cooperative reasoning with both views

The repository also includes **V2X-MoE**, a Qwen3-VL-based baseline with explicit view routing and three viewpoint-specific LoRA experts:
- `vs_expert`
- `is_expert`
- `co_expert`

## Important note on raw images
This repository **does not redistribute the original vehicle-side and infrastructure-side images from V2X-Seq-SPD**. Due to dataset licensing and redistribution constraints, users must download the raw images from the official V2X-Seq/V2X-Seq-SPD source separately and place them under the expected local directories before training or evaluation.

Please see:
- `data/README.md`
- `data/raw_external/README.md`
- `docs/REPRODUCE.md`

## Repository structure
```text
V2X-QA/
├── assets/
├── checkpoints/
│   ├── vs_expert/
│   ├── is_expert/
│   ├── co_expert/
│   ├── chat_template.jinja
│   ├── processor_config.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── data/
│   ├── train/
│   ├── test/
│   ├── raw_external/
│   ├── README.md
│   └── schema.md
├── docs/
│   ├── REPRODUCE.md
│   └── EVALUATION.md
├── model/
│   ├── train/
│   │   ├── v2x_moe_train_mcqa_qwen3_stage1.py
│   │   ├── v2x_moe_train_mcqa_qwen3_co_boost.py
│   │   └── v2x_moe_train_mcqa_qwen3_is_boost.py
│   └── eval/
│       └── v2x_moe_eval_mcqa_qwen3.py
├── .gitignore
├── CITATION.cff
├── environment.yml
├── LICENSE
├── requirements.txt
└── README.md
```

## Installation
### Option 1: Conda
```bash
conda env create -f environment.yml
conda activate v2x-qa
```

### Option 2: pip
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation
1. Put the released V2X-QA annotation JSONL files under:
```text
data/train/
data/test/
```
2. Download the original raw V2X-Seq-SPD images from the official source.
3. Place them under:
```text
data/raw_external/V2X-Seq-SPD-vehicle-side-image/
data/raw_external/V2X-Seq-SPD-infrastructure-side-image/
```

## Released checkpoints
The `checkpoints/` directory contains the released **final V2X-MoE adapters** and processor/tokenizer files needed for evaluation and reuse.

The released checkpoint layout is:
```text
checkpoints/
├── vs_expert/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── is_expert/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── co_expert/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── chat_template.jinja
├── processor_config.json
├── tokenizer.json
└── tokenizer_config.json
```

## Training
### Stage 1: joint MCQA training
```bash
python model/train/v2x_moe_train_mcqa_qwen3_stage1.py
```
This will save checkpoints to:
```text
outputs/stage1/
```

### Stage 2: CO-focused refinement
```bash
python model/train/v2x_moe_train_mcqa_qwen3_co_boost.py
```
This will save checkpoints to:
```text
outputs/stage2_co/
```

### Stage 3: IS-focused refinement
```bash
python model/train/v2x_moe_train_mcqa_qwen3_is_boost.py
```
This will save checkpoints to:
```text
outputs/stage3_is/
```

## Evaluation
To run evaluation with the released final checkpoint:
```bash
python model/eval/v2x_moe_eval_mcqa_qwen3.py
```
By default, the evaluation script reads from `checkpoints/` and writes outputs to:
```text
outputs/eval/
```

## What is released here
This repository is intended to release:
- V2X-QA annotation files (JSONL)
- V2X-MoE training and evaluation scripts
- Final released V2X-MoE LoRA adapters and tokenizer/processor files
- Documentation for reproduction and evaluation

## What is not redistributed here
This repository does **not** include:
- Original V2X-Seq-SPD raw images
- Any redistributed copy of the upstream base model weights

Users must comply with the licenses and usage terms of:
- the original V2X-Seq / V2X-Seq-SPD dataset source
- the upstream Qwen3-VL base model

## Citation
If you use this repository, dataset, or model, please cite the paper and the repository.

See `CITATION.cff` for the recommended repository citation.

### BibTeX (paper)
```bibtex
@article{you2026v2xqa,
  title   = {V2X-QA: A Comprehensive Reasoning Dataset and Benchmark for Multimodal Large Language Models in Autonomous Driving Across Ego, Infrastructure, and Cooperative Views},
  author  = {You, Junwei and Jiang, Zhuoyu and Li, Pei and Tang, Weizhe and Huang, Zilin and Gan, Rui and Liu, Jiaxi and Zhao, Yan and Chen, Sikai and Ran, Bin},
  journal = {arXiv preprint arXiv:REPLACE_WITH_ARXIV_ID},
  year    = {2026},
  url     = {REPLACE_WITH_ARXIV_LINK}
}
```

### BibTeX (repository)
```bibtex
@software{you2026v2xqa_repo,
  author = {You, Junwei and Jiang, Zhuoyu and Li, Pei and Tang, Weizhe and Huang, Zilin and Gan, Rui and Liu, Jiaxi and Zhao, Yan and Chen, Sikai and Ran, Bin},
  title  = {V2X-QA},
  year   = {2026},
  url    = {REPLACE_WITH_GITHUB_REPO_URL}
}
```

## Acknowledgment
This work builds on publicly available upstream resources, including V2X-Seq-SPD and Qwen3-VL. Please cite and follow the corresponding upstream licenses and terms.
