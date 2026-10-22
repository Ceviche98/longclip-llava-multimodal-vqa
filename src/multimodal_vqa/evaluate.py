"""
Multimodal VQA Evaluation Metrics, Post-Processing, and Inference Engines.

Implements:
- `extract_answer_letter`: Regex-backed parser extracting standalone MCQ choice letters
  (`A`, `B`, `C`, `D`) from raw generative outputs.
- `evaluation` & `compute_vqa_metrics`: Official exact-match VQA accuracy and per-option
  confusion/distribution breakdown.
- `get_prediction`: Deterministic greedy inference (`T=0.0`, left-padded batching,
  49-token `<image>` expansion).
- `get_ensemble_prediction`: Test-Time Compute / Self-Consistency majority-voting
  decoding across temperature `T` and `N` sampled reasoning paths (Experiment E5).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from tqdm import tqdm

from .dataset import NUM_VISION_PATCH_TOKENS, expand_image_tokens


def extract_answer_letter(text: Optional[str]) -> str:
    """
    Extract the first valid standalone answer letter (`A`, `B`, `C`, `D`, or `E`)
    from raw model output text.

    Handles exact single-token predictions (`'B'`), formatted strings (`'Answer: B'`,
    `'Option (C)'`), and noisy multi-token continuations before EOS discipline was enforced.
    Returns `'UNKNOWN'` if no valid letter is present.
    """
    if not text:
        return "UNKNOWN"

    cleaned = text.strip()
    if len(cleaned) == 1 and cleaned.upper() in ("A", "B", "C", "D", "E"):
        return cleaned.upper()

    # Look for standalone A-E bounded by word boundaries / punctuation
    match = re.search(r"\b([A-E])\b(?!\w)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: first occurrence of A-E
    match = re.search(r"([A-E])", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return "UNKNOWN"


def evaluation(gold_path: str, pred_path: str) -> float:
    """
    Official evaluation function computing exact-match VQA accuracy between
    ground-truth JSONL (`gold_path`) and predicted JSONL (`pred_path`).
    """
    gold_data: List[str] = []
    pred_data: List[str] = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                jdata = json.loads(line)
                gold_data.append(jdata["answer"])
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                jdata = json.loads(line)
                pred_data.append(jdata["answer"])
    if not gold_data:
        return 0.0
    n = min(len(gold_data), len(pred_data))
    return sum(1 for i in range(n) if gold_data[i] == pred_data[i]) / len(gold_data)


def compute_vqa_metrics(
    gold_answers: Sequence[str], pred_answers: Sequence[str]
) -> Dict[str, object]:
    """
    Compute comprehensive VQA evaluation metrics including exact-match accuracy,
    per-option accuracy, and predicted answer distribution.
    """
    if not gold_answers:
        return {"accuracy": 0.0, "total": 0, "correct": 0, "per_option_accuracy": {}}

    total = len(gold_answers)
    n = min(len(gold_answers), len(pred_answers))
    correct = sum(1 for i in range(n) if gold_answers[i] == pred_answers[i])
    accuracy = correct / total

    per_opt_totals: Counter = Counter()
    per_opt_correct: Counter = Counter()
    for i in range(n):
        g = gold_answers[i]
        p = pred_answers[i]
        per_opt_totals[g] += 1
        if g == p:
            per_opt_correct[g] += 1

    per_option_acc = {
        opt: (per_opt_correct[opt] / count if count > 0 else 0.0)
        for opt, count in sorted(per_opt_totals.items())
    }

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "per_option_accuracy": per_option_acc,
        "prediction_distribution": dict(Counter(pred_answers)),
    }


def get_prediction(
    model: torch.nn.Module,
    processor: object,
    batch_size: int,
    device: str,
    dataset: Dict[str, Sequence[Optional[str]]],
    max_new_tokens: int = 10,
) -> List[str]:
    """
    Run deterministic greedy batched inference (`do_sample=False`) on a VQA split.

    Enforces left-padding (`processor.tokenizer.padding_side = "left"`) during
    autoregressive generation and expands `<image>` into 49 visual patch tokens.
    """
    model.eval()
    predictions: List[str] = []

    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    image_paths = dataset["image_paths"]
    prompts = dataset["prompts"]

    print("Running inference...")
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Greedy Inference"):
        batch_images = []
        for path in image_paths[i : i + batch_size]:
            try:
                batch_images.append(Image.open(path).convert("RGB"))
            except Exception:
                batch_images.append(Image.new("RGB", (224, 224), color="black"))

        batch_prompts = prompts[i : i + batch_size]
        expanded_prompts = expand_image_tokens(batch_prompts, NUM_VISION_PATCH_TOKENS)

        text_inputs = processor.tokenizer(
            expanded_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        pixel_values = processor.image_processor(batch_images, return_tensors="pt")[
            "pixel_values"
        ].to(device)

        inputs = {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "pixel_values": pixel_values,
        }

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_texts = processor.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )

        for text in generated_texts:
            predictions.append(extract_answer_letter(text))

    processor.tokenizer.padding_side = original_padding_side
    return predictions


def get_ensemble_prediction(
    model: torch.nn.Module,
    processor: object,
    batch_size: int,
    device: str,
    data_dict: Dict[str, Sequence[Optional[str]]],
    temperature: float = 0.3,
    num_votes: int = 3,
) -> List[str]:
    """
    Experiment 5 (E5): Self-Consistency test-time compute ensemble inference.

    Samples `num_votes` responses per image at temperature `temperature` and selects
    the consensus answer via majority voting (`Counter(votes).most_common(1)`).
    When `temperature == 0.0` or `num_votes == 1`, defaults to deterministic greedy decoding.
    """
    model.eval()
    predictions: List[str] = []

    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    image_paths = data_dict["image_paths"]
    prompts = data_dict["prompts"]

    for i in tqdm(
        range(0, len(image_paths), batch_size),
        desc=f"Inference (T={temperature}, Votes={num_votes})",
    ):
        batch_img_paths = image_paths[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]

        images = []
        for p in batch_img_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(Image.new("RGB", (224, 224), color="black"))

        pixel_values = processor.image_processor(images, return_tensors="pt")[
            "pixel_values"
        ].to(device)
        expanded_prompts = expand_image_tokens(batch_prompts, NUM_VISION_PATCH_TOKENS)
        text_inputs = processor.tokenizer(
            expanded_prompts, return_tensors="pt", padding=True
        ).to(device)

        do_sample = temperature > 0.0 and num_votes > 1

        with torch.no_grad():
            outputs = model.generate(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=2,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_votes,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        input_len = text_inputs["input_ids"].shape[1]
        decoded_outputs = processor.tokenizer.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )

        for j in range(len(batch_prompts)):
            start_idx = j * num_votes
            end_idx = start_idx + num_votes
            img_responses = decoded_outputs[start_idx:end_idx]

            votes = []
            for resp in img_responses:
                valid_letters = [char for char in resp if char in ["A", "B", "C", "D"]]
                if valid_letters:
                    votes.append(valid_letters[0])
                else:
                    votes.append("A")

            most_common_vote = Counter(votes).most_common(1)[0][0]
            predictions.append(most_common_vote)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    processor.tokenizer.padding_side = original_padding_side
    return predictions


def write_answers_to_file(
    path: str,
    predictions: Sequence[str],
    answer_to_ids: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Write model predictions into the target JSONL file (`vqa_xxx_student_final.jsonl`).
    """
    if output_path is None:
        output_path = path.replace("with_image_ids.jsonl", "final.jsonl")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record, pred in zip(records, predictions):
            record["answer"] = pred
            out_f.write(json.dumps(record) + "\n")

    print(f"Predictions successfully written to {output_path}")
    return output_path
