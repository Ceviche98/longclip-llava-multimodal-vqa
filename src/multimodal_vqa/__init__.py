"""
LongCLIP + LLaVA Multimodal Visual Question Answering (VQA) Pipeline.

This package provides a modular, production-ready implementation of:
1. LongCLIP semantic image-caption alignment & positional embedding interpolation
   (extending standard 77-token CLIP context to 248 tokens).
2. Custom LLaVA multimodal architecture integrating LongCLIP-ViT-B/32 visual tokens
   with a Qwen3-0.6B causal language backbone via normalized multi-layer MLP projectors.
3. Parameter-Efficient Fine-Tuning (PEFT) via PiSSA (Principal Singular values and
   Singular vectors Adaptation) and selective vision encoder layer unfreezing.
4. Multimodal data loading, ColorJitter visual augmentation, dynamic MCQ option
   reshuffling, and self-consistency ensemble decoding.
"""

from .longclip_encoder import (
    CLIP_MODEL_ID,
    TEXT_MODEL_ID,
    interpolate_longclip_positional_embeddings,
    evaluate_image_caption_alignment,
    process_split,
    recover_image_id,
    LongCLIPAligner,
)
from .llava_arch import (
    LlavaMultiModalProjector,
    AdvancedMLPProjector,
    AdvancedMLPProjectorV2,
    build_custom_llava_model,
    apply_pissa_and_unfreeze,
)
from .dataset import (
    VQADataset,
    format_mcq_prompt,
    load_image_vqa_data,
    shuffle_options_in_batch,
    expand_image_tokens,
)
from .trainer import (
    VQAExperimentConfig,
    train_llava_epoch,
    run_experiment_pipeline,
)
from .evaluate import (
    extract_answer_letter,
    evaluation,
    compute_vqa_metrics,
    get_prediction,
    get_ensemble_prediction,
    write_answers_to_file,
)

__all__ = [
    "CLIP_MODEL_ID",
    "TEXT_MODEL_ID",
    "interpolate_longclip_positional_embeddings",
    "evaluate_image_caption_alignment",
    "process_split",
    "recover_image_id",
    "LongCLIPAligner",
    "LlavaMultiModalProjector",
    "AdvancedMLPProjector",
    "AdvancedMLPProjectorV2",
    "build_custom_llava_model",
    "apply_pissa_and_unfreeze",
    "VQADataset",
    "format_mcq_prompt",
    "load_image_vqa_data",
    "shuffle_options_in_batch",
    "expand_image_tokens",
    "VQAExperimentConfig",
    "train_llava_epoch",
    "run_experiment_pipeline",
    "extract_answer_letter",
    "evaluation",
    "compute_vqa_metrics",
    "get_prediction",
    "get_ensemble_prediction",
    "write_answers_to_file",
]
