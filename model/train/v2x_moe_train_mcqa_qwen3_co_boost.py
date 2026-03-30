#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""CO-focused refinement for V2X-MoE on Qwen3-VL.

Goal:
1) Improve CO1-CO4 accuracy efficiently.
2) Reuse an existing trained checkpoint instead of retraining everything.
3) Keep the original model architecture and adapter layout unchanged.
4) Improve CO prompting and reduce epoch time on a single RTX 4090.

Open-source note:
This repository does not redistribute the original V2X-Seq-SPD vehicle-side and
infrastructure-side images. Users must download the raw images from the official
source separately and update the paths below if needed.
"""

import os
import json
import glob
import math
import random
import shutil
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
RAW_EXTERNAL_ROOT = DATA_ROOT / "raw_external"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
OUTPUT_ROOT = REPO_ROOT / "outputs"

RAW_IMAGE_DATASET_NOTE = (
    "V2X-Seq-SPD raw vehicle-side and infrastructure-side images are not distributed "
    "with this repository. Please download them from the official source separately "
    "and place them under the expected directories, or update the corresponding paths "
    "in the Config section."
)


# ============================================================
# CONFIG
# ============================================================
class Config:
    mode = "train"

    # ----- resume / checkpoint -----
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    # change this to your best checkpoint before CO refinement
    base_checkpoint_dir = OUTPUT_ROOT / "stage1" / "epoch_4"
    output_dir = OUTPUT_ROOT / "stage2_co"

    # ----- training -----
    epochs = 2
    batch_size = 1
    learning_rate = 5e-5
    weight_decay = 0.01
    grad_accumulation_steps = 8
    warmup_ratio = 0.05
    max_grad_norm = 1.0
    seed = 42

    # ----- data -----
    train_jsonl_dir = DATA_ROOT / "train"
    image_root_vs = RAW_EXTERNAL_ROOT / "V2X-Seq-SPD-vehicle-side-image"
    image_root_is = RAW_EXTERNAL_ROOT / "V2X-Seq-SPD-infrastructure-side-image"
    # reduced from larger settings for faster CO-only refinement on 4090
    image_max_side = 560
    image_cache_size = 1024
    shuffle_options = True
    use_task_hint = True
    use_view_hint = True
    use_co_instruction = True
    co_task_balanced_sampling = True
    co_balance_alpha = 1.0

    # ----- runtime -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 2
    save_every_epoch = True
    log_every = 20
    use_flash_attention = True
    tf32 = True
    processor_max_pixels = 560 * 560


args = Config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v2x_moe_train_mcqa_qwen3_co_boost")

CO_TASKS = ["CO1", "CO2", "CO3", "CO4"]
COMMON_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


# ============================================================
# GLOBAL SPEED / REPRO
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_runtime() -> None:
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = True



def ensure_dir_exists(path, description: str, extra_hint: Optional[str] = None) -> None:
    path = Path(path)
    if not path.exists():
        message = f"{description} not found: {path}"
        if extra_hint:
            message += "\n" + str(extra_hint)
        raise FileNotFoundError(message)

# ============================================================
# utils
# ============================================================
def normalize_options(options: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for k in ["A", "B", "C", "D"]:
        if k in options:
            out[k] = str(options[k]).strip()
    return out


def find_image(root_dir: str, image_id: Optional[str]) -> Optional[str]:
    if not image_id:
        return None
    image_id = str(image_id)
    for ext in COMMON_IMAGE_EXTENSIONS:
        p = os.path.join(root_dir, image_id + ext)
        if os.path.exists(p):
            return p
    return None


@lru_cache(maxsize=Config.image_cache_size)
def load_image_cached(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img.copy()


def robust_load_image(path: str) -> Image.Image:
    try:
        return load_image_cached(path, args.image_max_side)
    except Exception:
        return Image.new("RGB", (224, 224), (0, 0, 0))


def shuffle_option_map(options: Dict[str, str], gold_letter: str) -> Tuple[Dict[str, str], str]:
    items = [(k, options[k]) for k in ["A", "B", "C", "D"] if k in options]
    random.shuffle(items)
    new_letters = ["A", "B", "C", "D"][:len(items)]
    new_options = {}
    new_gold = None
    for new_k, (old_k, text) in zip(new_letters, items):
        new_options[new_k] = text
        if old_k == gold_letter:
            new_gold = new_k
    if new_gold is None:
        raise ValueError(f"Failed to remap gold option: {gold_letter}")
    return new_options, new_gold


def build_co_prompt(task_id: str, question: str, options: Dict[str, str]) -> str:
    lines = []
    lines.append("You are solving a multiple-choice autonomous driving question.")
    if args.use_view_hint:
        lines.append("Image 1 is the vehicle-side view.")
        lines.append("Image 2 is the infrastructure-side view.")
    if args.use_task_hint:
        lines.append(f"Task ID: {task_id}.")
    if args.use_co_instruction:
        lines.append("Use both views jointly.")
        lines.append("Do not answer from only one view when the other view provides complementary evidence.")
        lines.append("When the two views differ in visibility, trust the view with clearer evidence for the queried target, but keep both views consistent.")
    lines.append("Choose the single best option.")
    lines.append("Output only one letter: A, B, C, or D.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Options:")
    for k in ["A", "B", "C", "D"]:
        if k in options:
            lines.append(f"{k}. {options[k]}")
    lines.append("Answer:")
    return "\n".join(lines)


def maybe_make_fused_adamw(params):
    try:
        return torch.optim.AdamW(
            params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            fused=torch.cuda.is_available(),
        )
    except TypeError:
        return torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)


def resolve_adapter_dir(checkpoint_root: str, adapter_name: str) -> str:
    candidates = [
        os.path.join(checkpoint_root, adapter_name),
        os.path.join(checkpoint_root, adapter_name, adapter_name),
        checkpoint_root if adapter_name == "vs_expert" else "",
    ]
    for cand in candidates:
        if cand and os.path.exists(os.path.join(cand, "adapter_config.json")):
            return cand
    raise FileNotFoundError(
        f"Could not find adapter_config.json for {adapter_name} under {checkpoint_root}"
    )


# ============================================================
# dataset
# ============================================================
class COOnlyTrainDataset(Dataset):
    def __init__(self, jsonl_dir: str, processor: AutoProcessor):
        self.samples: List[Dict] = []
        self.processor = processor
        self.root_vs = args.image_root_vs
        self.root_is = args.image_root_is

        logger.info("[CO-BOOST] Loading only CO1-CO4 jsonl files into RAM ...")
        self._load_and_cache(jsonl_dir)
        logger.info(f"[CO-BOOST] Loaded {len(self.samples)} valid CO training samples.")

        self.task_counts = Counter(s["task_id"] for s in self.samples)
        logger.info(f"[CO-BOOST] Task counts: {dict(self.task_counts)}")

    def _load_and_cache(self, jsonl_dir: str) -> None:
        jsonl_files = sorted(glob.glob(os.path.join(jsonl_dir, "CO*.jsonl")))
        if not jsonl_files:
            raise FileNotFoundError(f"No CO*.jsonl files found in: {jsonl_dir}")

        for path in jsonl_files:
            logger.info(f"Reading {os.path.basename(path)}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    task_id = str(row.get("task_id", "")).strip()
                    if task_id not in CO_TASKS:
                        continue

                    sample_id = str(row.get("sample_id", "")).strip()
                    question_id = str(row.get("question_id", "")).strip()
                    question = str(row.get("question", "")).strip()
                    options = normalize_options(row.get("options", {}) or {})
                    selected_option_id = str(row.get("selected_option_id", "")).strip().upper()
                    image_id_vs = row.get("image_id_vs")
                    image_id_is = row.get("image_id_is")

                    if not question or not options or selected_option_id not in options:
                        continue

                    p_vs = find_image(self.root_vs, image_id_vs)
                    p_is = find_image(self.root_is, image_id_is)
                    if not p_vs or not p_is:
                        continue

                    self.samples.append({
                        "sample_id": sample_id,
                        "task_id": task_id,
                        "question_id": question_id,
                        "question": question,
                        "options": options,
                        "gold_letter": selected_option_id,
                        # fixed order: Image 1 = VS, Image 2 = IS
                        "img_paths": [p_vs, p_is],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        data = self.samples[idx]
        options = dict(data["options"])
        gold_letter = data["gold_letter"]

        if args.shuffle_options:
            options, gold_letter = shuffle_option_map(options, gold_letter)

        img_vs = robust_load_image(data["img_paths"][0])
        img_is = robust_load_image(data["img_paths"][1])

        user_content = [
            {"type": "image", "image": img_vs},
            {"type": "image", "image": img_is},
        ]
        prompt = build_co_prompt(data["task_id"], data["question"], options)
        user_content.append({"type": "text", "text": prompt})

        prompt_messages = [{"role": "user", "content": user_content}]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        target_text = gold_letter
        eos_token = getattr(self.processor.tokenizer, "eos_token", None)
        if eos_token:
            target_text = target_text + eos_token

        full_text = prompt_text + target_text
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # qwen3-vl uses mm_token_type_ids, token_type_ids is not needed here
        inputs.pop("token_type_ids", None)

        batch = {k: v.squeeze(0) for k, v in inputs.items()}
        batch["task_id"] = data["task_id"]
        batch["sample_id"] = data["sample_id"]
        batch["question_id"] = data["question_id"]
        batch["view_type"] = "CO"

        labels = batch["input_ids"].clone()
        target_ids = self.processor.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        target_len = len(target_ids)
        if target_len <= 0 or target_len > labels.shape[0]:
            labels[:] = -100
        else:
            labels[:-target_len] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch


# ============================================================
# model / checkpoint io
# ============================================================
def setup_model_from_checkpoint():
    logger.info(f"Loading base model: {args.model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    attn_impl = "flash_attention_2" if args.use_flash_attention else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map=args.device,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    vision_tower = None
    if hasattr(model, "visual"):
        vision_tower = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        vision_tower = model.model.visual
    if vision_tower is not None:
        for p in vision_tower.parameters():
            p.requires_grad = False

    logger.info(f"Loading adapters from: {args.base_checkpoint_dir}")
    vs_dir = resolve_adapter_dir(args.base_checkpoint_dir, "vs_expert")
    is_dir = resolve_adapter_dir(args.base_checkpoint_dir, "is_expert")
    co_dir = resolve_adapter_dir(args.base_checkpoint_dir, "co_expert")

    model = PeftModel.from_pretrained(model, vs_dir, adapter_name="vs_expert")
    model.load_adapter(is_dir, adapter_name="is_expert")
    model.load_adapter(co_dir, adapter_name="co_expert")
    model.set_adapter("co_expert")

    # only train co_expert for targeted and faster refinement
    for name, p in model.named_parameters():
        p.requires_grad = ("co_expert" in name)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params (CO only): {trainable:,} / {total:,} ({100.0 * trainable / total:.4f}%)")
    return model


def _flatten_nested_adapter_dir(save_dir: str, adapter_name: str) -> None:
    direct_cfg = os.path.join(save_dir, "adapter_config.json")
    if os.path.exists(direct_cfg):
        return
    nested = os.path.join(save_dir, adapter_name)
    nested_cfg = os.path.join(nested, "adapter_config.json")
    if os.path.exists(nested_cfg):
        for fname in os.listdir(nested):
            src = os.path.join(nested, fname)
            dst = os.path.join(save_dir, fname)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        try:
            os.rmdir(nested)
        except OSError:
            pass


def save_single_adapter(model, save_dir: str, adapter_name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    model.set_adapter(adapter_name)
    try:
        model.save_pretrained(save_dir, selected_adapters=[adapter_name])
    except TypeError:
        model.save_pretrained(save_dir)
    _flatten_nested_adapter_dir(save_dir, adapter_name)


def save_checkpoint(model, processor, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    processor.save_pretrained(save_dir)
    # save all adapters for eval compatibility; vs/is remain unchanged
    save_single_adapter(model, os.path.join(save_dir, "vs_expert"), "vs_expert")
    save_single_adapter(model, os.path.join(save_dir, "is_expert"), "is_expert")
    save_single_adapter(model, os.path.join(save_dir, "co_expert"), "co_expert")
    logger.info(f"Saved checkpoint to {save_dir}")


# ============================================================
# dataloader helpers
# ============================================================
def build_task_balanced_sampler(dataset: COOnlyTrainDataset):
    if not args.co_task_balanced_sampling:
        return None
    counts = dataset.task_counts
    weights = []
    for sample in dataset.samples:
        c = counts[sample["task_id"]]
        weights.append((1.0 / c) ** args.co_balance_alpha)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# ============================================================
# training
# ============================================================
def prepare_grid_thw(grid_thw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if grid_thw is None:
        return None
    grid_thw = grid_thw.to(args.device, non_blocking=True)
    grid_thw = grid_thw.view(-1, grid_thw.shape[-1])
    if grid_thw.shape[-1] == 2:
        time_dim = torch.ones((grid_thw.shape[0], 1), dtype=grid_thw.dtype, device=grid_thw.device)
        grid_thw = torch.cat([time_dim, grid_thw], dim=-1)
    return grid_thw


def run_training(model, processor, train_loader):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = maybe_make_fused_adamw(trainable_params)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    warmup_steps = int(max_train_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    logger.info(
        f"Start CO-only refinement | epochs={args.epochs} | batch_size={args.batch_size} | "
        f"grad_acc={args.grad_accumulation_steps} | lr={args.learning_rate} | "
        f"updates/epoch={num_update_steps_per_epoch} | warmup_steps={warmup_steps}"
    )

    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"CO boost epoch {epoch + 1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(args.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)
            labels = batch["labels"].to(args.device, non_blocking=True)

            mm_token_type_ids = batch.get("mm_token_type_ids")
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids.to(args.device, non_blocking=True)

            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(args.device, non_blocking=True)

            grid_thw = prepare_grid_thw(batch.get("image_grid_thw"))

            model.set_adapter("co_expert")
            forward_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                labels=labels,
            )
            if mm_token_type_ids is not None:
                forward_kwargs["mm_token_type_ids"] = mm_token_type_ids

            outputs = model(**forward_kwargs)
            raw_loss = outputs.loss
            loss = raw_loss / args.grad_accumulation_steps
            loss.backward()
            running_loss += raw_loss.item()

            should_step = (step + 1) % args.grad_accumulation_steps == 0 or (step + 1) == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if (step + 1) % args.log_every == 0 or (step + 1) == len(train_loader):
                avg_loss = running_loss / min(args.log_every, (step % args.log_every) + 1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "task": batch["task_id"][0],
                })
                running_loss = 0.0

        if args.save_every_epoch:
            ckpt_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            save_checkpoint(model, processor, ckpt_dir)


# ============================================================
# main
# ============================================================
def main():
    setup_runtime()
    ensure_dir_exists(args.base_checkpoint_dir, "Base checkpoint directory for CO refinement")
    ensure_dir_exists(args.train_jsonl_dir, "Training annotation directory")
    ensure_dir_exists(args.image_root_vs, "Vehicle-side image root", RAW_IMAGE_DATASET_NOTE)
    ensure_dir_exists(args.image_root_is, "Infrastructure-side image root", RAW_IMAGE_DATASET_NOTE)
    os.makedirs(args.output_dir, exist_ok=True)

    processor_source = args.base_checkpoint_dir if os.path.exists(args.base_checkpoint_dir) else args.model_id
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        try:
            processor.image_processor.max_pixels = args.processor_max_pixels
        except Exception:
            pass

    model = setup_model_from_checkpoint()

    if args.mode in ["train", "all"]:
        train_ds = COOnlyTrainDataset(args.train_jsonl_dir, processor)
        sampler = build_task_balanced_sampler(train_ds)

        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = args.persistent_workers
            loader_kwargs["prefetch_factor"] = args.prefetch_factor

        if sampler is not None:
            train_loader = DataLoader(train_ds, sampler=sampler, shuffle=False, **loader_kwargs)
        else:
            train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

        run_training(model, processor, train_loader)


if __name__ == "__main__":
    main()
