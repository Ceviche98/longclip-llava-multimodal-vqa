#!/usr/bin/env python3
"""
CLI entrypoint for LongCLIP + LLaVA Multimodal VQA reconstruction and training.

Examples:
    # Step 1: Reconstruct validation & test image-to-MCQ mappings using LongCLIP (99.83% accuracy)
    python scripts/train_vqa.py --recover-images --data-dir ./data

    # Step 2: Run the winning E4-B optimization recipe (3-Layer LN-MLP + PiSSA r=32 + Unfreeze[-2:] + Jitter + Option Shuffle)
    python scripts/train_vqa.py --preset e4b --data-dir ./data
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from multimodal_vqa.longclip_encoder import CLIP_MODEL_ID, TEXT_MODEL_ID, recover_image_id
from multimodal_vqa.trainer import EXPERIMENT_PRESETS, run_experiment_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LongCLIP + Qwen3 Custom LLaVA Visual Question Answering Trainer"
    )
    parser.add_argument(
        "--recover-images",
        action="store_true",
        help="Run Task 1 LongCLIP contrastive image-caption alignment before training.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="e4b",
        choices=list(EXPERIMENT_PRESETS.keys()),
        help="Experiment configuration preset (task2_baseline, e1, e2a, e2b, e3a, e3b, e4a, e4b).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing VQA JSONL files and images/ folder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override peak learning rate.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default=CLIP_MODEL_ID,
        help="HuggingFace model ID for LongCLIP vision encoder.",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default=TEXT_MODEL_ID,
        help="HuggingFace model ID for Qwen3 causal language backbone.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recover_images:
        image_glob = os.path.join(args.data_dir, "images", "test_val", "*")
        recover_image_id(
            image_glob=image_glob,
            data_dir=args.data_dir,
            clip_model_id=args.clip_model,
        )

    config = EXPERIMENT_PRESETS[args.preset]
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr

    print(f"Launching LLaVA Experiment Preset '{config.name}' -> {config}")
    run_experiment_pipeline(
        config=config,
        data_dir=args.data_dir,
        clip_model_id=args.clip_model,
        text_model_id=args.text_model,
    )


if __name__ == "__main__":
    main()
