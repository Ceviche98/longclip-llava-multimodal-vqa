#!/usr/bin/env python3
"""
CLI entrypoint for evaluating VQA predictions, inspecting test-set distributions,
and running Experiment 5 (E5) Self-Consistency / Test-Time Compute grid search.

Examples:
    # Summarize the exported test-set predictions (5,538 MCQs)
    python scripts/evaluate_vqa.py --pred-path data/vqa_test_predictions.jsonl --summary-only

    # Evaluate predictions against a ground-truth teacher JSONL file
    python scripts/evaluate_vqa.py --gold-path vqa_val_teacher.jsonl --pred-path vqa_val_student_final.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from multimodal_vqa.evaluate import compute_vqa_metrics, evaluation
from multimodal_vqa.longclip_encoder import evaluate_image_caption_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Multimodal VQA Predictions and Alignment Accuracy"
    )
    parser.add_argument(
        "--gold-path",
        type=str,
        default=None,
        help="Path to ground-truth teacher JSONL file.",
    )
    parser.add_argument(
        "--pred-path",
        type=str,
        default="data/vqa_test_predictions.jsonl",
        help="Path to predicted student JSONL file.",
    )
    parser.add_argument(
        "--check-alignment",
        action="store_true",
        help="Also compute Task 1 LongCLIP image-caption alignment accuracy.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print prediction distribution summary without requiring a gold teacher file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Prediction file not found: {args.pred_path}")

    records = []
    with open(args.pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    ans_counts = Counter(r.get("answer", "UNKNOWN") for r in records)
    unique_images = len({r.get("image") for r in records if r.get("image")})
    print(f"Loaded {len(records)} MCQ predictions across {unique_images} unique images.")
    print("Predicted option distribution:")
    for opt in sorted(ans_counts.keys()):
        pct = 100.0 * ans_counts[opt] / max(len(records), 1)
        print(f"  Option {opt}: {ans_counts[opt]:4d} ({pct:5.2f}%)")

    if args.gold_path and not args.summary_only:
        if args.check_alignment:
            align_acc = evaluate_image_caption_alignment(args.gold_path, args.pred_path)
            print(f"Task 1 Image-Caption Alignment Accuracy: {align_acc * 100.0:.2f}%")

        gold_answers = []
        with open(args.gold_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    gold_answers.append(json.loads(line)["answer"])
        pred_answers = [r["answer"] for r in records]
        metrics = compute_vqa_metrics(gold_answers, pred_answers)
        official_acc = evaluation(args.gold_path, args.pred_path)
        print(f"Official VQA Exact-Match Accuracy: {official_acc * 100.0:.2f}%")
        print("Per-Option Accuracy:", metrics["per_option_accuracy"])


if __name__ == "__main__":
    main()
