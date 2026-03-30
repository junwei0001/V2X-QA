# Released Checkpoints

This directory stores the **released final V2X-MoE adapters** and processor/tokenizer files.

## Contents
The expected released checkpoint layout is:
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

## Notes
- `vs_expert`, `is_expert`, and `co_expert` are the three viewpoint-specific LoRA adapters used by V2X-MoE.
- The released checkpoint is intended for direct evaluation and reuse.
- Intermediate stage checkpoints are not required for public release unless you explicitly want to expose the full training trajectory.

## Upstream base model
The adapters in this repository are designed to work with the upstream base model:
- `Qwen/Qwen3-VL-8B-Instruct`

Users are responsible for complying with the base model license and access requirements.

## Loading behavior
The provided evaluation script expects this directory to contain:
- processor/tokenizer files at the root of `checkpoints/`
- one subdirectory for each adapter:
  - `vs_expert/`
  - `is_expert/`
  - `co_expert/`

## Naming note
Use `vs_expert`, **not** `va_expert`.
The provided scripts route to `vs_expert`, `is_expert`, and `co_expert` by name.
