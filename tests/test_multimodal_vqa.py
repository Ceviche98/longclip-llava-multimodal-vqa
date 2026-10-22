"""
Unit tests for the LongCLIP + LLaVA Multimodal VQA package (`multimodal_vqa`).

Tests cover:
1. Knowledge-preserving LongCLIP positional embedding interpolation (77 -> 248 tokens,
   preserving the first 20 token embeddings untouched).
2. Contrastive image-caption alignment similarity matrix and accuracy calculation.
3. Cross-modal MLP projector tensor shapes and gradient flow across all 3 architectures:
   - `LlavaMultiModalProjector` (Task 2 / E1)
   - `AdvancedMLPProjector` (E2-A: 2-Layer Pre-LN MLP)
   - `AdvancedMLPProjectorV2` (E2-B / E3 / E4 / E5: 3-Layer Pre-LN MLP)
4. VQA dataset prompt formatting, 49-token `<image>` expansion, dynamic MCQ option
   reshuffling (E4-B), and causal prompt-masking (`-100`).
5. Answer letter regex extraction (`extract_answer_letter`), exact-match VQA accuracy,
   and exported test prediction integrity (`data/vqa_test_predictions.jsonl`).
"""

from __future__ import annotations

import json
import os
import random
from types import SimpleNamespace

import pytest
import torch

from multimodal_vqa.dataset import (
    NUM_VISION_PATCH_TOKENS,
    VQADataset,
    expand_image_tokens,
    format_mcq_prompt,
    load_image_vqa_data,
    shuffle_options_in_batch,
)
from multimodal_vqa.evaluate import (
    compute_vqa_metrics,
    evaluation,
    extract_answer_letter,
    write_answers_to_file,
)
from multimodal_vqa.llava_arch import (
    AdvancedMLPProjector,
    AdvancedMLPProjectorV2,
    LlavaMultiModalProjector,
    build_projector,
)
from multimodal_vqa.longclip_encoder import (
    LongCLIPAligner,
    evaluate_image_caption_alignment,
    interpolate_longclip_positional_embeddings,
)
from multimodal_vqa.trainer import EXPERIMENT_PRESETS, build_masked_causal_labels


def _make_dummy_llava_config(vision_dim: int = 768, text_dim: int = 1024, act: str = "gelu"):
    return SimpleNamespace(
        vision_config=SimpleNamespace(hidden_size=vision_dim),
        text_config=SimpleNamespace(hidden_size=text_dim),
        projector_hidden_act=act,
    )


class TestLongCLIPPositionalInterpolation:
    def test_interpolate_2d_77_to_248_preserves_first_20(self):
        orig_pe = torch.randn(77, 512)
        stretched = interpolate_longclip_positional_embeddings(
            orig_pe, target_length=248, keep_first_n=20
        )
        assert stretched.shape == (248, 512)
        # First 20 positions must be preserved identically
        assert torch.allclose(stretched[:20], orig_pe[:20], atol=1e-6)
        # Final position (index 247) should match original final position (index 76) due to align_corners=True
        assert torch.allclose(stretched[-1], orig_pe[-1], atol=1e-5)

    def test_interpolate_3d_batch_dimension(self):
        orig_pe = torch.randn(2, 77, 768)
        stretched = interpolate_longclip_positional_embeddings(
            orig_pe, target_length=248, keep_first_n=20
        )
        assert stretched.shape == (2, 248, 768)
        assert torch.allclose(stretched[:, :20, :], orig_pe[:, :20, :], atol=1e-6)

    def test_cosine_alignment_matching(self, tmp_path):
        # Create orthogonal embeddings for 3 images and 3 captions
        img_embeds = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        txt_embeds = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 5.0], [3.0, 0.0, 0.0]])
        paths = ["images/test_val/img0.jpg", "images/test_val/img1.jpg", "images/test_val/img2.jpg"]

        matched, scores = LongCLIPAligner.match_captions_to_images(txt_embeds, img_embeds, paths)
        assert matched == [
            "images/test_val/img1.jpg",
            "images/test_val/img2.jpg",
            "images/test_val/img0.jpg",
        ]
        assert torch.allclose(scores, torch.ones(3), atol=1e-5)

        gold_file = tmp_path / "gold.jsonl"
        pred_file = tmp_path / "pred.jsonl"
        gold_records = [
            {"caption_id": "val_0", "image": "images/test_val/img0.jpg"},
            {"caption_id": "val_1", "image": "images/test_val/img1.jpg"},
        ]
        pred_records = [
            {"caption_id": "val_0", "image": "images/test_val/img0.jpg"},
            {"caption_id": "val_1", "image": "images/test_val/img2.jpg"},
        ]
        gold_file.write_text("\n".join(json.dumps(r) for r in gold_records) + "\n", encoding="utf-8")
        pred_file.write_text("\n".join(json.dumps(r) for r in pred_records) + "\n", encoding="utf-8")

        acc = evaluate_image_caption_alignment(str(gold_file), str(pred_file))
        assert acc == pytest.approx(0.5)


class TestCrossModalProjectors:
    @pytest.mark.parametrize("proj_type,cls", [
        ("baseline", LlavaMultiModalProjector),
        ("e2a", AdvancedMLPProjector),
        ("e2b", AdvancedMLPProjectorV2),
    ])
    def test_projector_output_shapes_and_gradients(self, proj_type, cls):
        cfg = _make_dummy_llava_config(vision_dim=768, text_dim=1024)
        projector = build_projector(proj_type, cfg)
        assert isinstance(projector, cls)

        # Batch of 4 images, 49 spatial patch tokens (7x7 ViT-B/32 without CLS), 768 hidden dim
        vision_tokens = torch.randn(4, NUM_VISION_PATCH_TOKENS, 768, requires_grad=True)
        projected = projector(vision_tokens)

        assert projected.shape == (4, 49, 1024)
        loss = projected.sum()
        loss.backward()
        assert vision_tokens.grad is not None
        assert vision_tokens.grad.shape == (4, 49, 768)


class TestVQADatasetAndAugmentations:
    def test_format_mcq_prompt_and_image_token_expansion(self):
        prompt = format_mcq_prompt(
            "What color is the frog?",
            ["Orange", "Blue", "Red", "Green"],
        )
        assert prompt.startswith("<image>\nAnswer with A, B, C, or D only.\n")
        assert "Question: What color is the frog?" in prompt
        assert "A. Orange\nB. Blue\nC. Red\nD. Green\nAnswer:\n" in prompt

        expanded = expand_image_tokens([prompt], num_tokens=49)[0]
        assert expanded.count("<image>") == 49

    def test_dynamic_option_reshuffling_preserves_correct_answer_semantics(self):
        prompt = format_mcq_prompt(
            "Which component is on the left?",
            ["Mounting hardware", "Main faucet body", "Valve seat", "Water supply lines"],
        )
        # Correct answer is B ("Main faucet body")
        prompts = [prompt for _ in range(20)]
        answers = ["B" for _ in range(20)]

        rng = random.Random(42)
        shuffled_prompts, shuffled_answers = shuffle_options_in_batch(prompts, answers, rng=rng)

        assert len(shuffled_prompts) == 20
        assert len(shuffled_answers) == 20
        # Ensure permutation actually varies the answer letters across the 20 trials
        assert len(set(shuffled_answers)) > 1

        for sp, sa in zip(shuffled_prompts, shuffled_answers):
            assert f"{sa}. Main faucet body\n" in sp

    def test_prompt_label_masking(self):
        # Batch of 2 sequences of length 6
        input_ids = torch.tensor([
            [10, 11, 12, 65, 151645, 151645],
            [10, 11, 12, 13, 66, 151645],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ])
        # First sample has prompt length 3; second sample has prompt length 4
        prompt_attention_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ])

        labels = build_masked_causal_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=prompt_attention_mask,
            pad_token_id=151645,
        )

        # Sample 0: positions 0, 1, 2 masked (prompt), pos 3 ('A'=65) & pos 4 (EOS=151645) kept, pos 5 masked (pad)
        assert labels[0].tolist() == [-100, -100, -100, 65, 151645, -100]
        # Sample 1: positions 0, 1, 2, 3 masked (prompt), pos 4 ('B'=66) & pos 5 (EOS=151645) kept
        assert labels[1].tolist() == [-100, -100, -100, -100, 66, 151645]


class TestEvaluationAndExport:
    @pytest.mark.parametrize("raw_text,expected", [
        ("B", "B"),
        (" D\nD\nD\nD\n", "D"),
        (" C \n B. Gray C. \n Black", "C"),
        ("Answer: A", "A"),
        ("Option (D) is correct", "D"),
        ("", "UNKNOWN"),
        (None, "UNKNOWN"),
    ])
    def test_extract_answer_letter(self, raw_text, expected):
        assert extract_answer_letter(raw_text) == expected

    def test_evaluation_and_write_answers(self, tmp_path):
        src_path = tmp_path / "vqa_val_student_with_image_ids.jsonl"
        gold_path = tmp_path / "vqa_val_teacher.jsonl"

        records = [
            {"image": "img1.jpg", "question": "Q1?", "options": ["o1", "o2", "o3", "o4"], "answer": None},
            {"image": "img2.jpg", "question": "Q2?", "options": ["o1", "o2", "o3", "o4"], "answer": None},
            {"image": "img3.jpg", "question": "Q3?", "options": ["o1", "o2", "o3", "o4"], "answer": None},
            {"image": "img4.jpg", "question": "Q4?", "options": ["o1", "o2", "o3", "o4"], "answer": None},
        ]
        gold_records = [
            {**records[0], "answer": "A"},
            {**records[1], "answer": "B"},
            {**records[2], "answer": "C"},
            {**records[3], "answer": "D"},
        ]

        src_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        gold_path.write_text("\n".join(json.dumps(r) for r in gold_records) + "\n", encoding="utf-8")

        loaded = load_image_vqa_data(str(src_path))
        ds = VQADataset(loaded)
        assert len(ds) == 4

        out_path = write_answers_to_file(str(src_path), ["A", "B", "C", "A"])
        acc = evaluation(str(gold_path), out_path)
        assert acc == pytest.approx(0.75)

        metrics = compute_vqa_metrics(["A", "B", "C", "D"], ["A", "B", "C", "A"])
        assert metrics["accuracy"] == pytest.approx(0.75)
        assert metrics["correct"] == 3
        assert metrics["total"] == 4

    def test_exported_test_predictions_file_integrity(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        pred_file = os.path.join(repo_root, "data", "vqa_test_predictions.jsonl")
        assert os.path.exists(pred_file), "data/vqa_test_predictions.jsonl must exist"

        with open(pred_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 5538
        first = json.loads(lines[0])
        assert set(first.keys()) >= {"image", "question", "options", "answer", "caption_id"}
        assert first["answer"] in {"A", "B", "C", "D"}

    def test_experiment_presets_exist(self):
        assert "e4b" in EXPERIMENT_PRESETS
        e4b = EXPERIMENT_PRESETS["e4b"]
        assert e4b.use_pissa is True
        assert e4b.lora_r == 32
        assert e4b.unfreeze_vision_layers == 2
        assert e4b.use_color_jitter is True
        assert e4b.use_option_shuffle is True
        assert e4b.epochs == 7
