"""
Multimodal VQA Dataset Loader, Prompt Templating, and Dynamic Augmentations.

Implements:
- Concise static MCQ prompt formatting (`format_mcq_prompt`).
- Manual `<image>` token expansion (`expand_image_tokens`) matching the 49 spatial
  patches produced by `LongCLIP-ViT-B-32` (`224x224` input, `32x32` patch size,
  `7x7 = 49` tokens after stripping CLS).
- Dynamic semantic option reshuffling (`shuffle_options_in_batch`) to eliminate
  small-LLM positional bias toward 'A' or 'C' (Experiment E4-B).
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Sequence, Tuple

from torch.utils.data import Dataset

DEFAULT_ANSWER_MAP: Dict[str, str] = {"A": "A", "B": "B", "C": "C", "D": "D"}
NUM_VISION_PATCH_TOKENS: int = 49


def format_mcq_prompt(question: str, options: Sequence[str]) -> str:
    """
    Construct the static, concise Multiple-Choice Question (MCQ) prompt for LLaVA.

    Chain-of-Thought (CoT) prompting was deliberately rejected on the 0.6B backbone
    to avoid reasoning trace hallucinations and attention dilution over visual tokens.
    """
    if len(options) < 4:
        raise ValueError(f"Expected 4 options (A, B, C, D), got {len(options)}")
    return (
        "<image>\nAnswer with A, B, C, or D only.\n"
        f"Question: {question}\n"
        "Options:\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        "Answer:\n"
    )


def expand_image_tokens(
    texts: Sequence[str], num_tokens: int = NUM_VISION_PATCH_TOKENS
) -> List[str]:
    """
    Expand each `<image>` placeholder into `num_tokens` space-separated `<image> `
    tokens so the tokenized sequence length matches the 49 patch embeddings from
    LongCLIP-ViT-B/32.
    """
    replacement = "<image> " * num_tokens
    return [t.replace("<image>", replacement) for t in texts]


def load_image_vqa_data(
    path: str, answer_to_ids: Optional[Dict[str, str]] = None
) -> Dict[str, List[Optional[str]]]:
    """
    Load image paths, formatted MCQ prompts, and target answer letters from a JSONL file.

    Args:
        path: Path to `.jsonl` file (`vqa_train_student.jsonl`, `vqa_val_student_with_image_ids.jsonl`, etc.).
        answer_to_ids: Optional dictionary mapping raw answer strings to target labels.

    Returns:
        Dictionary with keys `'image_paths'`, `'prompts'`, and `'answers'`.
    """
    if answer_to_ids is None:
        answer_to_ids = DEFAULT_ANSWER_MAP

    image_paths: List[Optional[str]] = []
    prompts: List[str] = []
    answers: List[Optional[str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            jdata = json.loads(line)
            image_paths.append(jdata.get("image"))
            question = jdata["question"]
            options = jdata["options"]
            answer = answer_to_ids.get(jdata.get("answer"), None)

            prompts.append(format_mcq_prompt(question, options))
            answers.append(answer)

    return {"image_paths": image_paths, "prompts": prompts, "answers": answers}


def shuffle_options_in_batch(
    prompts: Sequence[str],
    answers: Sequence[str],
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], List[str]]:
    """
    Dynamically parse and permute the A, B, C, D options within a training batch
    and update the ground-truth target letter accordingly (Experiment E4-B).

    Mitigates positional prior bias in compact (<1B) language models by forcing
    cross-modal attention between visual tokens and option text semantics.
    """
    rand = rng if rng is not None else random
    new_prompts: List[str] = []
    new_answers: List[str] = []

    for prompt, ans in zip(prompts, answers):
        try:
            head, tail = prompt.split("A. ", 1)
            opt_a, tail = tail.split("\nB. ", 1)
            opt_b, tail = tail.split("\nC. ", 1)
            opt_c, tail = tail.split("\nD. ", 1)
            opt_d, _ = tail.split("\nAnswer:\n", 1)

            options_list = [("A", opt_a), ("B", opt_b), ("C", opt_c), ("D", opt_d)]
            ans_clean = ans.strip()

            correct_text = next(text for letter, text in options_list if letter == ans_clean)
            opt_texts = [text for _, text in options_list]
            rand.shuffle(opt_texts)

            letters = ["A", "B", "C", "D"]
            new_options_str = ""
            new_ans = ""

            for i, text in enumerate(opt_texts):
                new_options_str += f"{letters[i]}. {text}\n"
                if text == correct_text:
                    new_ans = letters[i]

            new_prompt = head + new_options_str + "Answer:\n"
            new_prompts.append(new_prompt)
            new_answers.append(new_ans)
        except Exception:
            new_prompts.append(prompt)
            new_answers.append(ans)

    return new_prompts, new_answers


class VQADataset(Dataset):
    """PyTorch Dataset wrapping image paths, formatted MCQ prompts, and answers."""

    def __init__(self, data_dict: Dict[str, List[Optional[str]]]):
        self.image_paths = data_dict["image_paths"]
        self.prompts = data_dict["prompts"]
        self.answers = data_dict["answers"]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Optional[str], str, Optional[str]]:
        return self.image_paths[idx], self.prompts[idx], self.answers[idx]
