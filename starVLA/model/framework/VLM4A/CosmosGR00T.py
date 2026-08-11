# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
CosmosGR00T Framework — Cosmos-Reason2 VLM variant of GR00T.

Identical architecture to QwenGR00T (flow-matching DiT action head),
but uses Cosmos-Reason2 as the VLM backbone. Cosmos-Reason2 is built
on the Qwen3-VL architecture with physical-reasoning pretraining,
so it shares the same interface as other Qwen VLMs.
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.framework.VLM4A.gr00t_jepa import GR00TJEPAFrameworkMixin
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.action_model.tactile_jepa import REGION_GRIDS, VALID_IDX
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class CosmosGR00TDefaultConfig:
    """CosmosGR00T default parameters.

    Same structure as QwenGR00T but with world_model config section.
    """

    name: str = "CosmosGR00T"

    # === World Model backbone (Cosmos-Reason2) ===
    # Uses qwenvl key for backward compatibility with the shared interface
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/nvidia/Cosmos-Reason2-2B",
            "attn_implementation": "sdpa",
            "vl_hidden_dim": 2048,
        }
    )

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "future_action_window_size": 7,
            "action_horizon": 8,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "tactile_mode": "notac",
            "tactile_encoder_type": "mlp",
            "n_tactile_tokens": 8,
            "tactile_hidden_dim": 512,
            "tactile_num_heads": 8,
            "tactile_cnn_channels": 32,
            "tactile_cnn_pool": [2, 2],
            "tactile_cnn_coord_scale": 0.1,
            "tactile_raw_dim": 768,
            "tactile_valid_idx": list(VALID_IDX),
            "tactile_region_rows": [rows for rows, _ in REGION_GRIDS],
            "tactile_region_cols": [cols for _, cols in REGION_GRIDS],
            "tactile_cnn_coord": False,
            "dream_horizon": 4,
            "vision_horizon": 4,
            "dream_hidden_dim": 1024,
            "dream_state": False,
            "dream_vision": False,
            "ema_decay": 0.999,
            "tactile_dream_beta": 1.0,
            "lambda_tactile": 0.5,
            "lambda_state": 0.5,
            "lambda_vision": 0.5,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("CosmosGR00T")
class Cosmos_GR00T(GR00TJEPAFrameworkMixin, baseframework):
    """
    World-Model-for-Action framework (GR00T variant).

    Components:
      - Cosmos-Reason2 world model backbone (Qwen3-VL architecture)
      - Flow-matching (DiT) diffusion head for continuous action prediction

    The world model provides richer physical-reasoning representations
    compared to a standard VLM, while keeping the same hidden_states
    interface for downstream action prediction.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmosGR00TDefaultConfig, config)

        self.qwen_vl_interface = get_vlm_model(config=self.config)

        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self._init_gr00t_jepa()

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            return self._forward_gr00t_action(examples, last_hidden, backbone_attention_mask)

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self._predict_gr00t_action(
                examples, last_hidden, backbone_attention_mask
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
