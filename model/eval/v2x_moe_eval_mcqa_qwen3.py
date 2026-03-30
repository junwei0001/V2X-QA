#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calibration evaluation aligned with the generation-based V2X-MoE MCQA evaluation protocol.

Key alignment points
--------------------
1. Use the same SYSTEM_PROMPT and USER_PROMPT_TEMPLATE as the aligned generation-based eval script.
2. Keep the same system+user message structure and the same image note formatting.
3. Keep the same adapter routing (VS / IS / CO).
4. Prediction is generation-based for fairness/alignment.
5. Calibration confidence is computed from the first-step A/B/C/D probabilities
   under the exact same aligned prompt context.

Open-source note:
This repository does not redistribute the original V2X-Seq-SPD vehicle-side and
infrastructure-side images. Users must download the raw images from the official
source separately and update the paths below if needed.
"""

import os
import re
import json
import glob
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
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

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError as e:
    raise ImportError(
        "Qwen3VLForConditionalGeneration is not available in your current transformers. "
        "Please upgrade transformers to a version that includes Qwen3-VL."
    ) from e


class Config:
    test_jsonl_dir = DATA_ROOT / "test"
    img_root_vs = RAW_EXTERNAL_ROOT / "V2X-Seq-SPD-vehicle-side-image"
    img_root_is = RAW_EXTERNAL_ROOT / "V2X-Seq-SPD-infrastructure-side-image"

    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    checkpoint_dir = CHECKPOINT_ROOT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_max_side = 560
    image_cache_size = 1024
    processor_max_pixels = 560 * 560
    use_flash_attention = True
    tf32 = True

    max_new_tokens = 5
    do_sample = False
    num_beams = 1

    max_samples_per_task = None
    n_bins = 12
    min_plot_bin_count = 8  # only for plotting; ECE/Brier remain unchanged

    save_predictions = False
    predictions_path = OUTPUT_ROOT / "eval" / "eval_calib_predictions_aligned.jsonl"
    save_summary_json = False
    summary_json_path = OUTPUT_ROOT / "eval" / "eval_calib_summary_aligned.json"
    save_plot_png = True
    plot_png_path = OUTPUT_ROOT / "eval" / "mode_conditioned_reliability_aligned.png"


args = Config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("v2x_moe_eval_mcqa_qwen3_calib_aligned")

TASK_ORDER = [
    "VS1", "VS2", "VS3", "VS4",
    "IS1", "IS2", "IS3", "IS4",
    "CO1", "CO2", "CO3", "CO4",
]
COMMON_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
LETTER_SET = {"A", "B", "C", "D"}

SYSTEM_PROMPT = (
    "You are answering a multiple-choice autonomous driving question. "
    "Use only the provided image evidence and the question/options. "
    "Return exactly one uppercase letter only: A or B or C or D. "
    "Do not output any other words, punctuation, or explanation."
)

USER_PROMPT_TEMPLATE = """Task: {task_id}
Viewpoint: {viewpoint}

Image evidence:
{image_note}

Question:
{question}

Options:
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

Return exactly one uppercase letter only: A, B, C, or D.
"""

CHOICE_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def setup_runtime() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = True


def infer_view(task_id: str, image_id_vs: Optional[str], image_id_is: Optional[str]) -> str:
    if isinstance(task_id, str):
        if task_id.startswith("VS"):
            return "VS"
        if task_id.startswith("IS"):
            return "IS"
        if task_id.startswith("CO"):
            return "CO"
    if image_id_vs and image_id_is:
        return "CO"
    if image_id_vs:
        return "VS"
    if image_id_is:
        return "IS"
    return "UNKNOWN"


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


def resolve_adapter_dir(checkpoint_root: str, adapter_name: str) -> str:
    candidates = [
        checkpoint_root,
        os.path.join(checkpoint_root, adapter_name),
        os.path.join(checkpoint_root, adapter_name, adapter_name),
    ]
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "adapter_config.json")):
            return cand
    raise FileNotFoundError(
        f"Could not find adapter_config.json for {adapter_name} under {checkpoint_root}. "
        f"Checked: {candidates}"
    )


def maybe_to_device(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.to(args.device, non_blocking=True)


def prepare_grid_thw(grid_thw: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if grid_thw is None:
        return None
    grid_thw = grid_thw.to(args.device, non_blocking=True)
    grid_thw = grid_thw.view(-1, grid_thw.shape[-1])
    if grid_thw.shape[-1] == 2:
        time_dim = torch.ones((grid_thw.shape[0], 1), dtype=grid_thw.dtype, device=grid_thw.device)
        grid_thw = torch.cat([time_dim, grid_thw], dim=-1)
    return grid_thw


def parse_choice_from_text(text: str) -> str:
    cleaned = text.strip().upper()
    if cleaned in LETTER_SET:
        return cleaned
    m = CHOICE_RE.search(cleaned)
    if m:
        choice = m.group(1).upper()
        if choice in LETTER_SET:
            return choice
    raise ValueError(f"Failed to parse model choice from response text: {text[:200]}")


def build_user_text(sample: Dict) -> str:
    if sample["view"] == "VS":
        image_note = "One image is provided: vehicle-side (ego) view."
    elif sample["view"] == "IS":
        image_note = "One image is provided: infrastructure-side (RSU) view."
    elif sample["view"] == "CO":
        image_note = (
            "Two images are provided in order: "
            "(1) vehicle-side (ego) view, "
            "(2) infrastructure-side (RSU) view."
        )
    else:
        raise ValueError(f"Unknown viewpoint: {sample['view']}")

    return USER_PROMPT_TEMPLATE.format(
        task_id=sample["task_id"],
        viewpoint=sample["view"],
        image_note=image_note,
        question=sample["question"],
        opt_a=sample["options"]["A"],
        opt_b=sample["options"]["B"],
        opt_c=sample["options"]["C"],
        opt_d=sample["options"]["D"],
    )


def append_jsonl(path: str, row: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_and_processor():
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
    model.config.use_cache = True
    model.eval()

    processor_source = args.checkpoint_dir if os.path.exists(args.checkpoint_dir) else args.model_id
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        try:
            processor.image_processor.max_pixels = args.processor_max_pixels
        except Exception:
            pass

    logger.info(f"Loading adapters from: {args.checkpoint_dir}")
    vs_dir = resolve_adapter_dir(args.checkpoint_dir, "vs_expert")
    is_dir = resolve_adapter_dir(args.checkpoint_dir, "is_expert")
    co_dir = resolve_adapter_dir(args.checkpoint_dir, "co_expert")

    model = PeftModel.from_pretrained(model, vs_dir, adapter_name="vs_expert")
    model.load_adapter(is_dir, adapter_name="is_expert")
    model.load_adapter(co_dir, adapter_name="co_expert")
    model.eval()

    tokenizer = processor.tokenizer
    letter_token_ids = {}
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(
                f"Expected single token for candidate letter {letter}, got token ids {ids}. "
                f"This aligned calibration script assumes one-token letters."
            )
        letter_token_ids[letter] = ids[0]

    logger.info(f"Letter token ids: {letter_token_ids}")
    return model, processor, letter_token_ids


def load_test_samples(jsonl_dir: str) -> List[Dict]:
    paths = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"No jsonl files found in: {jsonl_dir}")

    samples = []
    counts = {task: 0 for task in TASK_ORDER}

    for path in paths:
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
                if task_id not in TASK_ORDER:
                    continue
                if args.max_samples_per_task is not None and counts[task_id] >= args.max_samples_per_task:
                    continue

                question = str(row.get("question", "")).strip()
                options = normalize_options(row.get("options", {}) or {})
                gold = str(row.get("selected_option_id", "")).strip().upper()
                image_id_vs = row.get("image_id_vs")
                image_id_is = row.get("image_id_is")
                view = infer_view(task_id, image_id_vs, image_id_is)

                if not question or gold not in options or view == "UNKNOWN":
                    continue

                img_paths = []
                if view in ["VS", "CO"]:
                    p = find_image(args.img_root_vs, image_id_vs)
                    if p:
                        img_paths.append(p)
                if view in ["IS", "CO"]:
                    p = find_image(args.img_root_is, image_id_is)
                    if p:
                        img_paths.append(p)
                if not img_paths:
                    continue

                samples.append({
                    "task_id": task_id,
                    "question_id": str(row.get("question_id", "")).strip(),
                    "sample_id": str(row.get("sample_id", "")).strip(),
                    "view": view,
                    "question": question,
                    "options": options,
                    "gold": gold,
                    "img_paths": img_paths,
                    "image_id_vs": image_id_vs,
                    "image_id_is": image_id_is,
                })
                counts[task_id] += 1

    logger.info(f"Loaded {len(samples)} test samples.")
    logger.info(f"Per-task counts: {counts}")
    return samples


@torch.no_grad()
def evaluate_single_sample(model, processor, letter_token_ids: Dict[str, int], sample: Dict):
    images = [robust_load_image(p) for p in sample["img_paths"]]

    system_message = {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    user_content = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": build_user_text(sample)})
    user_message = {"role": "user", "content": user_content}
    messages = [system_message, user_message]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    proc = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = proc["input_ids"].to(args.device, non_blocking=True)
    attention_mask = proc["attention_mask"].to(args.device, non_blocking=True)
    pixel_values = maybe_to_device(proc.get("pixel_values"))
    image_grid_thw = prepare_grid_thw(proc.get("image_grid_thw"))
    token_type_ids = maybe_to_device(proc.get("token_type_ids"))
    mm_token_type_ids = maybe_to_device(proc.get("mm_token_type_ids"))

    view = sample["view"]
    if view == "VS":
        model.set_adapter("vs_expert")
    elif view == "IS":
        model.set_adapter("is_expert")
    elif view == "CO":
        model.set_adapter("co_expert")
    else:
        raise ValueError(f"Unknown view type: {view}")

    shared_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if pixel_values is not None:
        shared_kwargs["pixel_values"] = pixel_values
    if image_grid_thw is not None:
        shared_kwargs["image_grid_thw"] = image_grid_thw
    if token_type_ids is not None:
        shared_kwargs["token_type_ids"] = token_type_ids
    if mm_token_type_ids is not None:
        shared_kwargs["mm_token_type_ids"] = mm_token_type_ids

    generated = model.generate(
        **shared_kwargs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        use_cache=True,
    )
    generated_ids = generated[0, input_ids.shape[1]:]
    decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    pred = parse_choice_from_text(decoded)

    outputs = model(**shared_kwargs)
    next_token_logits = outputs.logits[:, -1, :].float().squeeze(0)

    letters = ["A", "B", "C", "D"]
    score_vec = torch.tensor(
        [next_token_logits[letter_token_ids[l]].item() for l in letters],
        dtype=torch.float32
    )
    prob_vec = torch.softmax(score_vec, dim=0).cpu().numpy()

    prob_dict = {l: float(prob_vec[i]) for i, l in enumerate(letters)}
    score_dict = {l: float(score_vec[i].item()) for i, l in enumerate(letters)}
    conf = float(np.max(prob_vec))

    return pred, decoded, conf, prob_dict, score_dict


def compute_ece(confs: List[float], accs: List[float], n_bins: int = 15) -> Tuple[float, List[Tuple[float, float, float, int]]]:
    confs = np.asarray(confs, dtype=np.float32)
    accs = np.asarray(accs, dtype=np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []

    N = len(confs)
    if N == 0:
        return 0.0, bin_stats

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs > lo) & (confs <= hi)

        cnt = int(mask.sum())
        if cnt == 0:
            continue

        bin_conf = float(confs[mask].mean())
        bin_acc = float(accs[mask].mean())
        w = cnt / N
        ece += abs(bin_acc - bin_conf) * w
        bin_stats.append(((lo + hi) / 2.0, bin_conf, bin_acc, cnt))

    return float(ece), bin_stats


def safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else 0.0


def build_summary(stats: Dict[str, Dict[str, int]], calib: Dict[str, Dict[str, List[float]]], n_bins: int) -> Dict:
    summary = {
        "accuracy": {},
        "calibration": {},
    }
    for key in ["VS", "IS", "CO", "ALL"] + TASK_ORDER:
        c = stats[key]["c"]
        t = stats[key]["t"]
        acc = (100.0 * c / t) if t > 0 else 0.0
        summary["accuracy"][key] = {"correct": c, "total": t, "acc": acc}

    for key in ["VS", "IS", "CO", "ALL"]:
        ece, bins = compute_ece(calib[key]["conf"], calib[key]["acc"], n_bins=n_bins)
        summary["calibration"][key] = {
            "ece": ece,
            "brier": safe_mean(calib[key]["brier"]),
            "n": len(calib[key]["conf"]),
            "bin_stats": [
                {"bin_center": bc, "avg_conf": ac, "avg_acc": aa, "count": cnt}
                for (bc, ac, aa, cnt) in bins
            ],
        }
    return summary


def print_summary(stats: Dict[str, Dict[str, int]], calib: Dict[str, Dict[str, List[float]]], n_bins: int) -> None:
    print("\n================ Evaluation Summary ================")
    for key in ["VS", "IS", "CO", "ALL"] + TASK_ORDER:
        c = stats[key]["c"]
        t = stats[key]["t"]
        acc = (100.0 * c / t) if t > 0 else 0.0
        print(f"{key:>4} | correct={c:>5} | total={t:>5} | acc={acc:>7.2f}%")
    print("====================================================\n")

    print("================ Calibration Summary ================")
    print(f"{'View':>4} | {'ECE':>8} | {'Brier':>8} | {'N':>6}")
    for key in ["VS", "IS", "CO", "ALL"]:
        ece, _ = compute_ece(calib[key]["conf"], calib[key]["acc"], n_bins=n_bins)
        brier = safe_mean(calib[key]["brier"])
        n = len(calib[key]["conf"])
        print(f"{key:>4} | {ece:>8.4f} | {brier:>8.4f} | {n:>6}")
    print("====================================================\n")


def plot_calibration_curves(calib: Dict[str, Dict[str, List[float]]], n_bins: int = 15, save_path: Optional[str] = None) -> None:
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    palette = {
        "VS": "#1F4E79",
        "IS": "#7C3AED",
        "CO": "#B91C1C",
    }

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0
    bw = bins[1] - bins[0]

    def bin_stats_local(confs, accs):
        confs = np.asarray(confs, dtype=np.float32)
        accs = np.asarray(accs, dtype=np.float32)
        avg_conf = np.full(n_bins, np.nan, dtype=np.float32)
        avg_acc = np.full(n_bins, np.nan, dtype=np.float32)
        counts = np.zeros(n_bins, dtype=np.int32)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            m = (confs >= lo) & (confs <= hi) if i == 0 else (confs > lo) & (confs <= hi)
            counts[i] = int(m.sum())
            if counts[i] > 0:
                avg_conf[i] = float(confs[m].mean())
                avg_acc[i] = float(accs[m].mean())
        return avg_conf, avg_acc, counts

    def compute_ece_from_bins(avg_conf, avg_acc, counts):
        valid = ~np.isnan(avg_conf)
        if valid.sum() == 0:
            return 0.0
        N = counts.sum()
        return float(np.nansum(np.abs(avg_acc - avg_conf) * (counts / max(1, N))))

    def smooth1d(x, k=5):
        if k <= 1:
            return x
        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(x, kernel, mode="same")

    fig = plt.figure(figsize=(9.2, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axd = fig.add_subplot(gs[1], sharex=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.8, alpha=0.65, color="#6B7280")
    offsets = {"VS": -0.25, "IS": 0.0, "CO": 0.25}
    bar_w = bw * 0.28

    for v in ["VS", "IS", "CO"]:
        avg_conf, avg_acc, counts = bin_stats_local(calib[v]["conf"], calib[v]["acc"])
        valid_all = ~np.isnan(avg_conf)
        valid_plot = valid_all & (counts >= args.min_plot_bin_count)
        if valid_plot.sum() == 0:
            continue

        # Keep ECE exactly as before: computed from all non-empty bins, not only plotted bins.
        ece = compute_ece_from_bins(avg_conf, avg_acc, counts)
        xbars = centers + offsets[v] * bw
        ax.bar(
            xbars[valid_plot],
            avg_acc[valid_plot],
            width=bar_w,
            color=palette[v],
            alpha=0.18,
            edgecolor=palette[v],
            linewidth=1.0,
        )
        ax.plot(
            avg_conf[valid_plot],
            avg_acc[valid_plot],
            color=palette[v],
            linewidth=2.6,
            marker="o",
            markersize=5.8,
            label=f"{v} (ECE={ece:.3f})",
        )

        hist, _ = np.histogram(np.asarray(calib[v]["conf"], dtype=np.float32), bins=bins, density=False)
        h = smooth1d(hist.astype(np.float32), k=5)
        if h.max() > 0:
            h = h / h.max()

        axd.plot(centers, h, color=palette[v], linewidth=2.2, alpha=0.95)
        axd.fill_between(centers, h, 0, color=palette[v], alpha=0.10)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Mode-Conditioned Reliability (MCQA)")
    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.35)
    ax.legend(loc="upper left", frameon=True, edgecolor="#D1D5DB")
    plt.setp(ax.get_xticklabels(), visible=False)

    axd.set_ylim(0, 1.05)
    axd.set_xlabel("Confidence")
    axd.set_ylabel("Density")
    axd.grid(True, linestyle=":", linewidth=0.9, alpha=0.25)

    for a in [ax, axd]:
        for sp in a.spines.values():
            sp.set_color("#9CA3AF")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    setup_runtime()
    ensure_dir_exists(args.checkpoint_dir, "Released checkpoint directory")
    ensure_dir_exists(args.test_jsonl_dir, "Test annotation directory")
    ensure_dir_exists(args.img_root_vs, "Vehicle-side image root", RAW_IMAGE_DATASET_NOTE)
    ensure_dir_exists(args.img_root_is, "Infrastructure-side image root", RAW_IMAGE_DATASET_NOTE)

    model, processor, letter_token_ids = load_model_and_processor()
    samples = load_test_samples(args.test_jsonl_dir)

    if args.save_predictions:
        os.makedirs(os.path.dirname(args.predictions_path), exist_ok=True)
        if os.path.exists(args.predictions_path):
            os.remove(args.predictions_path)

    stats = {k: {"c": 0, "t": 0} for k in ["VS", "IS", "CO", "ALL"] + TASK_ORDER}
    calib = {k: {"conf": [], "acc": [], "brier": []} for k in ["VS", "IS", "CO", "ALL"]}

    print("\n>>> Start aligned generation-based MCQA calibration evaluation.")
    for sample in tqdm(samples):
        pred, decoded, conf, prob_dict, score_dict = evaluate_single_sample(
            model, processor, letter_token_ids, sample
        )
        gold = sample["gold"]
        ok = int(pred == gold)

        view = sample["view"]
        task = sample["task_id"]
        stats[view]["t"] += 1
        stats[view]["c"] += ok
        stats[task]["t"] += 1
        stats[task]["c"] += ok
        stats["ALL"]["t"] += 1
        stats["ALL"]["c"] += ok

        calib[view]["conf"].append(conf)
        calib[view]["acc"].append(float(ok))
        calib["ALL"]["conf"].append(conf)
        calib["ALL"]["acc"].append(float(ok))

        letters = ["A", "B", "C", "D"]
        p = np.array([prob_dict[c] for c in letters], dtype=np.float32)
        y = np.zeros(4, dtype=np.float32)
        if gold in letters:
            y[letters.index(gold)] = 1.0
        brier = float(np.sum((p - y) ** 2))
        calib[view]["brier"].append(brier)
        calib["ALL"]["brier"].append(brier)

        if args.save_predictions:
            append_jsonl(args.predictions_path, {
                "task_id": task,
                "question_id": sample["question_id"],
                "sample_id": sample["sample_id"],
                "view": view,
                "gold_choice": gold,
                "pred_choice": pred,
                "raw_output": decoded,
                "correct": bool(ok),
                "confidence": conf,
                "probabilities": prob_dict,
                "scores": score_dict,
                "image_id_vs": sample["image_id_vs"],
                "image_id_is": sample["image_id_is"],
            })

    print_summary(stats, calib, args.n_bins)

    summary = build_summary(stats, calib, args.n_bins)
    if args.save_summary_json:
        os.makedirs(os.path.dirname(args.summary_json_path), exist_ok=True)
        with open(args.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summary json to: {args.summary_json_path}")

    plot_calibration_curves(
        calib,
        n_bins=args.n_bins,
        save_path=args.plot_png_path if args.save_plot_png else None,
    )


if __name__ == "__main__":
    main()
