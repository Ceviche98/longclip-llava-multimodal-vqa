"""
Stage-1 & Stage-2 Visual Instruction Tuning and Task 3 Optimization Strategies.

Supports all experimental configurations documented in the ablation study:
- `task2_baseline`: Standard 2-layer MLP projector, no EOS token (67.18% lenient val accuracy,
  suffers from repetitive continuation noise).
- `e1_strict_eos`: Appends `<|im_end|>` EOS token to target labels + prompt label masking (`-100`)
  so loss is computed strictly on the single answer letter + EOS (61.61% honest baseline).
- `e2a_mlp2`: 2-layer pre-LayerNorm MLP projector + Cosine LR schedule with 3% warmup (64.15%).
- `e2b_mlp3`: 3-layer pre-LayerNorm MLP projector + Cosine LR schedule with 3% warmup (64.76%).
- `e3a_pissa`: E2-B + PiSSA SVD-initialized low-rank adaptation (`r=32, alpha=64`) across all
  7 Qwen3 attention/MLP projections (68.06%).
- `e3b_unfreeze`: E3-A + selective unfreezing of the final 2 LongCLIP vision encoder blocks (68.39%).
- `e4a_color_jitter`: E3-B + `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)` (68.55%).
- `e4b_combined`: E4-A + dynamic MCQ option permutation during batch construction across 7 epochs
  (**69.12% peak validation accuracy**, **67.62% official hidden test accuracy / Cohort Rank #18**).
"""

from __future__ import annotations

import copy
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .dataset import (
    DEFAULT_ANSWER_MAP,
    NUM_VISION_PATCH_TOKENS,
    VQADataset,
    expand_image_tokens,
    load_image_vqa_data,
    shuffle_options_in_batch,
)
from .evaluate import evaluation, get_prediction, write_answers_to_file
from .llava_arch import apply_pissa_and_unfreeze, build_custom_llava_model
from .longclip_encoder import CLIP_MODEL_ID, TEXT_MODEL_ID


@dataclass
class VQAExperimentConfig:
    """Configuration dataclass encapsulating Task 2 and Task 3 optimization routes."""

    name: str = "e4b_combined"
    projector_type: str = "e2b"  # 'baseline', 'e2a', 'e2b'
    append_eos_token: bool = True
    use_cosine_schedule: bool = True
    warmup_ratio: float = 0.03
    use_pissa: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    unfreeze_vision_layers: int = 2
    use_color_jitter: bool = True
    use_option_shuffle: bool = True
    epochs: int = 7
    batch_size: int = 14
    inference_batch_size: int = 4
    learning_rate: float = 2e-5
    seed: int = 0
    output_dir: str = "E4_Combined"


EXPERIMENT_PRESETS: Dict[str, VQAExperimentConfig] = {
    "task2_baseline": VQAExperimentConfig(
        name="task2_baseline",
        projector_type="baseline",
        append_eos_token=False,
        use_cosine_schedule=False,
        use_pissa=False,
        unfreeze_vision_layers=0,
        use_color_jitter=False,
        use_option_shuffle=False,
        epochs=3,
        batch_size=16,
        output_dir="Task2_Baseline",
    ),
    "e1": VQAExperimentConfig(
        name="e1",
        projector_type="baseline",
        append_eos_token=True,
        use_cosine_schedule=False,
        use_pissa=False,
        unfreeze_vision_layers=0,
        use_color_jitter=False,
        use_option_shuffle=False,
        epochs=3,
        batch_size=14,
        output_dir="E1",
    ),
    "e2a": VQAExperimentConfig(
        name="e2a",
        projector_type="e2a",
        append_eos_token=True,
        use_cosine_schedule=True,
        use_pissa=False,
        unfreeze_vision_layers=0,
        use_color_jitter=False,
        use_option_shuffle=False,
        epochs=5,
        batch_size=14,
        output_dir="E2",
    ),
    "e2b": VQAExperimentConfig(
        name="e2b",
        projector_type="e2b",
        append_eos_token=True,
        use_cosine_schedule=True,
        use_pissa=False,
        unfreeze_vision_layers=0,
        use_color_jitter=False,
        use_option_shuffle=False,
        epochs=5,
        batch_size=14,
        output_dir="E2_V2",
    ),
    "e3a": VQAExperimentConfig(
        name="e3a",
        projector_type="e2b",
        append_eos_token=True,
        use_cosine_schedule=True,
        use_pissa=True,
        unfreeze_vision_layers=0,
        use_color_jitter=False,
        use_option_shuffle=False,
        epochs=5,
        batch_size=14,
        output_dir="E3a",
    ),
    "e3b": VQAExperimentConfig(
        name="e3b",
        projector_type="e2b",
        append_eos_token=True,
        use_cosine_schedule=True,
        use_pissa=True,
        unfreeze_vision_layers=2,
        use_color_jitter=False,
        use_option_shuffle=False,
        epochs=5,
        batch_size=14,
        output_dir="E3b",
    ),
    "e4a": VQAExperimentConfig(
        name="e4a",
        projector_type="e2b",
        append_eos_token=True,
        use_cosine_schedule=True,
        use_pissa=True,
        unfreeze_vision_layers=2,
        use_color_jitter=True,
        use_option_shuffle=False,
        epochs=5,
        batch_size=14,
        output_dir="E4a",
    ),
    "e4b": VQAExperimentConfig(
        name="e4b",
        projector_type="e2b",
        append_eos_token=True,
        use_cosine_schedule=True,
        use_pissa=True,
        unfreeze_vision_layers=2,
        use_color_jitter=True,
        use_option_shuffle=True,
        epochs=7,
        batch_size=14,
        output_dir="E4_Combined",
    ),
}


def build_masked_causal_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    pad_token_id: Optional[int],
) -> torch.Tensor:
    """
    Create causal language modeling labels with padding tokens and prompt tokens
    masked to `-100` so cross-entropy loss is computed exclusively on the target
    answer letter and `<|im_end|>` EOS token.
    """
    labels = input_ids.clone()
    if pad_token_id is not None:
        labels[attention_mask == 0] = -100

    batch_size = input_ids.shape[0]
    for j in range(batch_size):
        prompt_length = int((prompt_attention_mask[j] == 1).sum().item())
        labels[j, :prompt_length] = -100

    return labels


def train_llava_epoch(
    model: torch.nn.Module,
    processor: object,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[object],
    config: VQAExperimentConfig,
    device: str,
    epoch_idx: int,
) -> float:
    """Execute one training epoch with optional ColorJitter and dynamic option reshuffling."""
    model.train()
    color_jitter = (
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        if config.use_color_jitter
        else None
    )
    eos_token = processor.tokenizer.eos_token if config.append_eos_token else ""
    total_loss = 0.0
    num_steps = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}/{config.epochs}")
    for batch_img_paths, batch_prompts, batch_answers in loop:
        images = []
        for p in batch_img_paths:
            try:
                img = Image.open(p).convert("RGB")
                if color_jitter is not None:
                    img = color_jitter(img)
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", (224, 224), color="black"))

        if config.use_option_shuffle:
            active_prompts, active_answers = shuffle_options_in_batch(
                batch_prompts, batch_answers
            )
        else:
            active_prompts, active_answers = list(batch_prompts), list(batch_answers)

        full_texts = [
            f"{p} {a}{eos_token}" for p, a in zip(active_prompts, active_answers)
        ]
        pixel_values = processor.image_processor(images, return_tensors="pt")[
            "pixel_values"
        ].to(device)

        expanded_texts = expand_image_tokens(full_texts, NUM_VISION_PATCH_TOKENS)
        expanded_prompts = expand_image_tokens(active_prompts, NUM_VISION_PATCH_TOKENS)

        text_inputs = processor.tokenizer(
            expanded_texts, return_tensors="pt", padding=True
        ).to(device)
        prompt_inputs = processor.tokenizer(
            expanded_prompts, return_tensors="pt", padding=True
        )

        labels = build_masked_causal_labels(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
            prompt_attention_mask=prompt_inputs.attention_mask,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

        inputs = {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "pixel_values": pixel_values,
        }

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        total_loss += float(loss.item())
        num_steps += 1
        lr_val = scheduler.get_last_lr()[0] if scheduler is not None else config.learning_rate
        loop.set_postfix(loss=loss.item(), lr=lr_val)

    return total_loss / max(num_steps, 1)


def run_experiment_pipeline(
    config: VQAExperimentConfig,
    data_dir: str = ".",
    clip_model_id: str = CLIP_MODEL_ID,
    text_model_id: str = TEXT_MODEL_ID,
) -> float:
    """
    End-to-end training, per-epoch validation checkpointing, and final test prediction
    generation for any `VQAExperimentConfig` preset.
    """
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    model_dir = os.path.join(config.output_dir, "model")
    val_dir = os.path.join(config.output_dir, "validation")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = build_custom_llava_model(
        clip_model_id=clip_model_id,
        text_model_id=text_model_id,
        projector_type=config.projector_type,
        device=device,
    )

    if config.use_pissa or config.unfreeze_vision_layers > 0:
        model = apply_pissa_and_unfreeze(
            model=model,
            use_pissa=config.use_pissa,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            unfreeze_vision_layers=config.unfreeze_vision_layers,
        )
        model.to(device)
    else:
        for param in model.model.vision_tower.parameters():
            param.requires_grad = False

    train_path = os.path.join(data_dir, "vqa_train_student.jsonl")
    val_path = os.path.join(data_dir, "vqa_val_student_with_image_ids.jsonl")
    test_path = os.path.join(data_dir, "vqa_test_student_with_image_ids.jsonl")
    teacher_val_path = os.path.join(data_dir, "vqa_val_teacher.jsonl")

    train_data = load_image_vqa_data(train_path, DEFAULT_ANSWER_MAP)
    train_dataset = VQADataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )

    scheduler = None
    if config.use_cosine_schedule:
        num_training_steps = config.epochs * len(train_loader)
        num_warmup_steps = int(config.warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    best_val_acc = 0.0
    best_model_state = None
    best_model_path = os.path.join(model_dir, "best_model.pth")

    for epoch in range(config.epochs):
        train_llava_epoch(
            model=model,
            processor=processor,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            epoch_idx=epoch,
        )

        print(f"\nEvaluating Epoch {epoch + 1}...")
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            fresh_val_data = load_image_vqa_data(val_path, DEFAULT_ANSWER_MAP)
            temp_val_preds = get_prediction(
                model, processor, config.inference_batch_size, device, fresh_val_data
            )
            temp_val_path = os.path.join(val_dir, f"vqa_val_epoch_{epoch + 1}.jsonl")
            write_answers_to_file(
                val_path, temp_val_preds, DEFAULT_ANSWER_MAP, output_path=temp_val_path
            )
            current_val_acc = evaluation(teacher_val_path, temp_val_path)

        print(f"Epoch {epoch + 1} Validation Accuracy: {current_val_acc * 100.0:.2f}%")
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, best_model_path)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    final_val_data = load_image_vqa_data(val_path, DEFAULT_ANSWER_MAP)
    final_test_data = load_image_vqa_data(test_path, DEFAULT_ANSWER_MAP)

    final_val_preds = get_prediction(
        model, processor, config.inference_batch_size, device, final_val_data
    )
    final_test_preds = get_prediction(
        model, processor, config.inference_batch_size, device, final_test_data
    )

    final_val_out = os.path.join(val_dir, "vqa_val_student_final.jsonl")
    final_test_out = os.path.join(val_dir, "vqa_test_student_final.jsonl")
    write_answers_to_file(val_path, final_val_preds, DEFAULT_ANSWER_MAP, output_path=final_val_out)
    write_answers_to_file(test_path, final_test_preds, DEFAULT_ANSWER_MAP, output_path=final_test_out)

    shutil.copy(final_test_out, os.path.join(data_dir, "vqa_test_student_final.jsonl"))
    final_eval_acc = evaluation(teacher_val_path, final_val_out)
    print(f"FINAL Official validation accuracy on val set: {final_eval_acc * 100.0:.2f}%")
    return final_eval_acc
