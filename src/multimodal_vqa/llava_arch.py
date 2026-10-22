"""
Custom LLaVA Multimodal Architecture & Cross-Modal Projectors.

Integrates the `AlpachinoNLP/LongCLIP-ViT-B-32` vision encoder (extracting 49 spatial
patch tokens from the penultimate transformer layer `vision_feature_layer=-2`) with
the `Qwen/Qwen3-0.6B` causal language model backbone.

Includes:
- `LlavaMultiModalProjector`: Standard 2-layer GELU MLP connector (Task 2 & E1).
- `AdvancedMLPProjector`: 2-layer pre-LayerNorm GELU MLP connector (Experiment E2-A).
- `AdvancedMLPProjectorV2`: 3-layer pre-LayerNorm GELU MLP connector (Experiments E2-B, E3, E4, E5).
- `apply_pissa_and_unfreeze`: PiSSA SVD-initialized low-rank adaptation (`r=32, alpha=64`)
  on all 7 Qwen3 attention/MLP projections plus selective unfreezing of the final
  2 LongCLIP vision transformer blocks.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)
from transformers.activations import ACT2FN

from .longclip_encoder import CLIP_MODEL_ID, TEXT_MODEL_ID


class LlavaMultiModalProjector(nn.Module):
    """
    Task 2 & Experiment 1 (E1) baseline 2-layer MLP projector:
    `Linear(vision_dim -> text_dim) -> GELU -> Linear(text_dim -> text_dim)`.
    """

    def __init__(self, config: LlavaConfig):
        super().__init__()
        vision_dim = config.vision_config.hidden_size
        text_dim = config.text_config.hidden_size
        act_name = getattr(config, "projector_hidden_act", "gelu")
        self.linear_1 = nn.Linear(vision_dim, text_dim, bias=True)
        self.act = ACT2FN[act_name]
        self.linear_2 = nn.Linear(text_dim, text_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class AdvancedMLPProjector(nn.Module):
    """
    Experiment E2-A 2-layer pre-LayerNorm MLP projector:
    `LayerNorm(vision_dim) -> Linear(vision_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> text_dim)`.

    Pre-LayerNorm stabilizes gradient magnitudes flowing from raw LongCLIP patch tokens
    into the Qwen3 embedding space.
    """

    def __init__(self, config: LlavaConfig):
        super().__init__()
        vision_dim = config.vision_config.hidden_size
        text_dim = config.text_config.hidden_size
        hidden_dim = text_dim
        act_name = getattr(config, "projector_hidden_act", "gelu")

        self.proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            ACT2FN[act_name],
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AdvancedMLPProjectorV2(nn.Module):
    """
    Experiment E2-B / E3 / E4 / E5 3-layer pre-LayerNorm MLP projector:
    `LayerNorm(vision_dim) -> Linear(vision_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> text_dim)`.

    Deepening the cross-modal connector to 3 layers with LayerNorm and cosine warmup
    raised validation accuracy from 61.61% (E1) to 64.76% (E2-B) even with a frozen LLM.
    """

    def __init__(self, config: LlavaConfig):
        super().__init__()
        vision_dim = config.vision_config.hidden_size
        hidden_dim = config.text_config.hidden_size
        text_dim = config.text_config.hidden_size
        act_name = getattr(config, "projector_hidden_act", "gelu")
        act_fn = ACT2FN[act_name]

        self.proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def build_projector(projector_type: str, config: LlavaConfig) -> nn.Module:
    """Instantiate the requested multimodal projector variant."""
    projector_type = projector_type.lower()
    if projector_type in ("baseline", "e1", "linear_mlp"):
        return LlavaMultiModalProjector(config)
    if projector_type in ("e2a", "mlp_2layer_ln"):
        return AdvancedMLPProjector(config)
    if projector_type in ("e2b", "v2", "mlp_3layer_ln"):
        return AdvancedMLPProjectorV2(config)
    raise ValueError(f"Unknown projector_type: {projector_type}")


def build_custom_llava_model(
    clip_model_id: str = CLIP_MODEL_ID,
    text_model_id: str = TEXT_MODEL_ID,
    projector_type: str = "e2b",
    device: str = "cpu",
) -> Tuple[LlavaForConditionalGeneration, LlavaProcessor]:
    """
    Construct the custom LLaVA model by stitching `AlpachinoNLP/LongCLIP-ViT-B-32`
    and `Qwen/Qwen3-0.6B` into a `LlavaForConditionalGeneration` shell.

    Key architectural decisions:
    - `patch_size = 32`, `vision_feature_layer = -2`, `vision_feature_select_strategy = 'default'`
      (strips the CLS token from the 50 ViT-B/32 tokens, yielding exactly 7x7 = 49 spatial patch tokens).
    - Adds `<image>` special token to the Qwen3 tokenizer and resizes token embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    image_token_str = "<image>"
    if image_token_str not in tokenizer.get_vocab():
        tokenizer.add_tokens([image_token_str], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids(image_token_str)
    image_processor = CLIPImageProcessor.from_pretrained(clip_model_id)

    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=32,
        vision_feature_select_strategy="default",
        num_additional_image_tokens=0,
    )
    processor.patch_size = 32
    processor.vision_feature_select_strategy = "default"
    processor.num_additional_image_tokens = 0

    text_config = AutoConfig.from_pretrained(text_model_id)
    vision_config = CLIPVisionConfig.from_pretrained(clip_model_id)
    vision_config.output_hidden_states = True

    llava_config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        ignore_index=-100,
        image_token_index=image_token_index,
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
    )

    model = LlavaForConditionalGeneration(llava_config)
    llm = AutoModelForCausalLM.from_pretrained(text_model_id)
    model.lm_head.load_state_dict(llm.lm_head.state_dict())
    model.model.language_model.load_state_dict(llm.model.state_dict(), strict=False)
    model.resize_token_embeddings(len(tokenizer))

    vision_model = CLIPVisionModel.from_pretrained(clip_model_id)
    model.model.vision_tower.load_state_dict(vision_model.state_dict(), strict=False)
    model.model.multi_modal_projector = build_projector(projector_type, llava_config)
    model.to(device)

    return model, processor


def apply_pissa_and_unfreeze(
    model: nn.Module,
    use_pissa: bool = True,
    lora_r: int = 32,
    lora_alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    unfreeze_vision_layers: int = 2,
) -> nn.Module:
    """
    Configure parameter-efficient fine-tuning (PEFT) via PiSSA on the language model
    and selective top-layer unfreezing on the LongCLIP vision encoder (Experiments E3-A & E3-B).

    Args:
        model: Instantiated `LlavaForConditionalGeneration` (or compatible wrapper).
        use_pissa: If True, wraps `model.model.language_model` with PEFT `LoraConfig`
            using `init_lora_weights="pissa"`.
        lora_r: Rank of SVD-initialized low-rank matrices (default `32`).
        lora_alpha: Scaling factor (default `64`).
        target_modules: Attention and MLP projection names in Qwen3.
        unfreeze_vision_layers: Number of terminal LongCLIP transformer encoder blocks
            to unfreeze (`0` for E1/E2/E3-A, `2` for E3-B/E4/E5).
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    # 1. Language model PEFT configuration
    if use_pissa:
        from peft import LoraConfig, get_peft_model

        for param in model.model.language_model.parameters():
            param.requires_grad = False

        pissa_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            init_lora_weights="pissa",
        )
        model.model.language_model = get_peft_model(model.model.language_model, pissa_config)

    # 2. Vision encoder selective unfreezing
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

    if unfreeze_vision_layers > 0:
        if hasattr(model.model.vision_tower, "vision_model"):
            vision_layers = model.model.vision_tower.vision_model.encoder.layers
        else:
            vision_layers = model.model.vision_tower.encoder.layers

        for layer in vision_layers[-unfreeze_vision_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    # 3. Always train the cross-modal projector
    for param in model.model.multi_modal_projector.parameters():
        param.requires_grad = True

    return model
