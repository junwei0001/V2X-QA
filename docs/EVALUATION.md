# Evaluation Guide

This document summarizes the evaluation setup used by the released V2X-MoE script.

## Script
```bash
python model/eval/v2x_moe_eval_mcqa_qwen3.py
```

## Inputs
The evaluation script expects:
1. public V2X-QA test JSONL files under:
```text
data/test/
```
2. external raw V2X-Seq-SPD images under:
```text
data/raw_external/V2X-Seq-SPD-vehicle-side-image/
data/raw_external/V2X-Seq-SPD-infrastructure-side-image/
```
3. released final checkpoint under:
```text
checkpoints/
```

## Checkpoint structure
The script expects:
```text
checkpoints/
├── vs_expert/
├── is_expert/
├── co_expert/
├── chat_template.jinja
├── processor_config.json
├── tokenizer.json
└── tokenizer_config.json
```

## View routing
During evaluation, the script uses explicit hard routing based on the viewpoint inferred from the task and image fields:
- `VS*` tasks route to `vs_expert`
- `IS*` tasks route to `is_expert`
- `CO*` tasks route to `co_expert`

## Prediction protocol
The evaluation script uses generation-based MCQA prediction:
- it constructs a system prompt and user prompt
- it generates a short answer
- it parses the generated answer into one of `A/B/C/D`

## Calibration and reliability
The script additionally computes confidence-based reliability metrics for V2X-MoE.
These are intended to complement, not replace, the main benchmark accuracy results.

The script includes:
- accuracy summary by view and by task
- ECE computation from confidence bins
- Brier score computation using the 4-way option probability vector and one-hot target vector
- reliability curve plotting

## Default outputs
By default, the script saves a reliability plot to:
```text
outputs/eval/
```
Optional outputs can also include:
- prediction JSONL
- summary JSON

## Public benchmark note
If public test annotations include `selected_option_id`, users can reproduce evaluation locally.
If you later choose to maintain a hidden-label benchmark, you should keep the private answer file outside the public repository.
