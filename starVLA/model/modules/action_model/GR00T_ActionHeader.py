# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT


from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT
from starVLA.model.modules.action_model.tactile_jepa import (
    RAW_DIM,
    REGION_GRIDS,
    VALID_IDX,
    DreamHead,
    TactileEncoder,
    build_ema_teacher,
    ema_update,
    jepa_loss,
)

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config

        # ------------------------------------------------------------------
        # DiT architecture selection
        #   action_model_type: "DiT-B" | "DiT-L"
        #     DiT-B → input_embedding_dim=768,  heads=12, head_dim=64
        #     DiT-L → input_embedding_dim=1536, heads=32, head_dim=48
        #   diffusion_model_cfg overrides/extends the base DiT shape.
        #   In particular, diffusion_model_cfg.cross_attention_dim MUST be
        #   set by the framework to match the VLM hidden size BEFORE calling
        #   get_action_model(), e.g.:
        #       cfg.framework.action_model.diffusion_model_cfg.cross_attention_dim
        #           = vlm.model.config.hidden_size
        # ------------------------------------------------------------------
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)

        # ------------------------------------------------------------------
        # Action horizon (chunk length sent to the DiT)
        #   Single source of truth: `action_horizon` (e.g. 8).
        #   Legacy YAMLs that only provide `future_action_window_size` are
        #   normalised to `action_horizon` upstream by
        #   `share_tools.apply_config_compat`, so this code never touches
        #   the legacy alias.
        # ------------------------------------------------------------------
        self.action_horizon = int(config.action_horizon)

        # ------------------------------------------------------------------
        # Action / state dimensions
        #   action_dim: DoF of the robot action (e.g. 7 for 6-DoF + gripper)
        #   state_dim:  proprioception dimension; set to 0/None to disable
        #               the state_encoder branch entirely.
        # ------------------------------------------------------------------
        self.action_dim = config.action_dim

        # ------------------------------------------------------------------
        # Inference denoising steps
        #   num_inference_timesteps: Euler steps during predict_action().
        #   Typically 4–10; fewer = faster but less accurate.
        # ------------------------------------------------------------------
        self.num_inference_timesteps = config.num_inference_timesteps

        # ------------------------------------------------------------------
        # hidden_size: intermediate MLP width for state_encoder / action_decoder.
        #   Decoupled from input_embedding_dim so you can use a smaller hidden
        #   for the MLP without changing the DiT latent size.
        # ------------------------------------------------------------------
        self.hidden_size = config.hidden_size

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # ------------------------------------------------------------------
        # future_tokens: learnable query tokens prepended before the action
        #   sequence so the DiT has dedicated "planning" slots.
        #   num_target_vision_tokens controls how many such tokens are added.
        # ------------------------------------------------------------------
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Positional embedding over the action sequence
        #   add_pos_embed: whether to add sinusoidal-style learned PE
        #   max_seq_len:   max supported action sequence length
        # ------------------------------------------------------------------
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # ------------------------------------------------------------------
        # Flow-matching noise schedule (Beta distribution)
        #   noise_beta_alpha / noise_beta_beta: Beta(α, β) shape params.
        #   noise_s: upper-clip of the sampled value so t ∈ [0, noise_s].
        #   num_timestep_buckets: discretise continuous t into N buckets for
        #     the timestep encoder inside DiT.
        # ------------------------------------------------------------------
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

        self.tactile_mode = str(config.get("tactile_mode", "notac")).lower()
        if self.tactile_mode not in {"notac", "input", "dream"}:
            raise ValueError(f"Unknown tactile_mode {self.tactile_mode!r}")
        self.use_tactile = self.tactile_mode != "notac"
        self.use_tactile_dream = self.tactile_mode == "dream"
        self.dream_state = self.use_tactile_dream and bool(config.get("dream_state", False))
        self.dream_vision = self.use_tactile_dream and bool(config.get("dream_vision", False))
        self.dream_horizon = int(config.get("dream_horizon", 4))
        self.vision_horizon = int(config.get("vision_horizon", self.dream_horizon))
        self.ema_decay = float(config.get("ema_decay", 0.999))
        self.jepa_beta = float(config.get("tactile_dream_beta", 1.0))
        self.lambda_tactile = float(config.get("lambda_tactile", 0.5))
        self.lambda_state = float(config.get("lambda_state", 0.5))
        self.lambda_vision = float(config.get("lambda_vision", 0.5))

        if self.use_tactile:
            if not bool(self.model.config.interleave_self_attention):
                raise ValueError(
                    "tactile fusion requires diffusion_model_cfg.interleave_self_attention=true"
                )
            tactile_hidden_dim = int(config.get("tactile_hidden_dim", 512))
            region_rows = tuple(config.get("tactile_region_rows", [rows for rows, _ in REGION_GRIDS]))
            region_cols = tuple(config.get("tactile_region_cols", [cols for _, cols in REGION_GRIDS]))
            if len(region_rows) != len(region_cols):
                raise ValueError("tactile_region_rows and tactile_region_cols must have equal lengths")
            encoder_type = str(config.get("tactile_encoder_type", "mlp"))
            if encoder_type == "cnn" and bool(config.get("tactile_cnn_coord", False)):
                encoder_type = "coord"
            self.tactile_encoder = TactileEncoder(
                embed_dim=self.input_embedding_dim,
                hidden_dim=tactile_hidden_dim,
                num_tokens=int(config.get("n_tactile_tokens", 8)),
                num_heads=int(config.get("tactile_num_heads", 8)),
                encoder_type=encoder_type,
                cnn_channels=int(config.get("tactile_cnn_channels", 32)),
                cnn_pool=tuple(config.get("tactile_cnn_pool", (2, 2))),
                coord_scale=float(config.get("tactile_cnn_coord_scale", 0.1)),
                raw_dim=int(config.get("tactile_raw_dim", RAW_DIM)),
                valid_idx=tuple(config.get("tactile_valid_idx", VALID_IDX)),
                region_grids=tuple(zip(region_rows, region_cols, strict=True)),
            )

        if self.use_tactile_dream:
            dream_hidden_dim = int(config.get("dream_hidden_dim", self.hidden_size))
            trunk_dim = int(self.model.config.output_dim)
            self.tactile_target_encoder = build_ema_teacher(self.tactile_encoder)
            self.tactile_dream_head = DreamHead(
                trunk_dim,
                self.input_embedding_dim,
                self.dream_horizon,
                dream_hidden_dim,
            )
            if self.dream_state:
                if self.state_encoder is None:
                    raise ValueError("dream_state requires a non-zero state_dim")
                self.state_target_encoder = build_ema_teacher(self.state_encoder)
                self.state_dream_head = DreamHead(
                    trunk_dim,
                    self.input_embedding_dim,
                    self.dream_horizon,
                    dream_hidden_dim,
                )
            if self.dream_vision:
                self.vision_target_dim = int(
                    config.get("vision_target_dim", config.diffusion_model_cfg.cross_attention_dim)
                )
                self.vision_dream_head = DreamHead(
                    trunk_dim,
                    self.vision_target_dim,
                    self.vision_horizon,
                    dream_hidden_dim,
                )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        tactile: torch.Tensor = None,
        future_state: torch.Tensor = None,
        future_vision_target: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        tactile_future_mask: torch.Tensor = None,
        state_future_mask: torch.Tensor = None,
        vision_future_mask: torch.Tensor = None,
        encoder_attention_mask=None,
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        if action_mask is not None:
            action_mask = action_mask.to(device=actions.device, dtype=actions.dtype)
            if action_mask.shape != actions.shape:
                raise ValueError(
                    f"Action mask shape {tuple(action_mask.shape)} does not match "
                    f"actions {tuple(actions.shape)}"
                )
            actions = actions * action_mask
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if action_mask is not None:
            noise = noise * action_mask
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # embed state
        state_features = self.state_encoder(state) if state is not None else None
        if state_features is not None and state_features.ndim == 2:
            state_features = state_features.unsqueeze(1)

        tactile_features = None
        if self.use_tactile:
            if tactile is None:
                raise ValueError("tactile fusion training requires a current tactile frame")
            if tactile.ndim == 2:
                tactile_current = tactile
            elif tactile.ndim == 3:
                tactile_current = tactile[:, 0]
            else:
                raise ValueError(f"Expected tactile [B, 256] or [B, T, 256], got {tactile.shape}")
            tactile_features = self.tactile_encoder(tactile_current)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        token_groups = []
        if state_features is not None:
            token_groups.append(state_features)
        token_groups.append(future_tokens)
        tactile_slice = None
        if tactile_features is not None:
            tactile_start = sum(group.shape[1] for group in token_groups)
            token_groups.append(tactile_features)
            tactile_slice = slice(tactile_start, tactile_start + tactile_features.shape[1])
        token_groups.append(action_features)
        sa_embs = torch.cat(token_groups, dim=1)

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_error = (pred_actions - velocity) ** 2
        if action_mask is None:
            action_loss = action_error.mean()
        else:
            action_loss = (action_error * action_mask).sum() / action_mask.sum().clamp_min(1.0)
        if not self.use_tactile_dream:
            return action_loss

        expected_tactile_shape = (self.dream_horizon + 1, self.tactile_encoder.raw_dim)
        if tactile.ndim != 3 or tactile.shape[1:] != expected_tactile_shape:
            raise ValueError(
                "tactile dream requires "
                f"[B, {expected_tactile_shape[0]}, {self.tactile_encoder.raw_dim}], "
                f"got {tactile.shape}"
            )
        if tactile_slice is None:
            raise RuntimeError("Tactile token slice was not constructed")
        tactile_trunk = model_output[:, tactile_slice].mean(dim=1)
        with torch.no_grad():
            tactile_target = self.tactile_target_encoder.encode_pooled(
                tactile[:, 1 : 1 + self.dream_horizon]
            )
        losses = {
            "action_loss": action_loss,
            "tactile_loss": jepa_loss(
                self.tactile_dream_head(tactile_trunk),
                tactile_target,
                beta=self.jepa_beta,
                mask=tactile_future_mask,
            ),
        }

        if self.dream_state:
            expected = self.dream_horizon
            if future_state is None or future_state.ndim != 3 or future_state.shape[1] != expected:
                shape = None if future_state is None else tuple(future_state.shape)
                raise ValueError(f"state-JEPA requires [B, {expected}, D], got {shape}")
            with torch.no_grad():
                state_target = self.state_target_encoder(future_state)
            losses["state_jepa_loss"] = jepa_loss(
                self.state_dream_head(tactile_trunk),
                state_target,
                beta=self.jepa_beta,
                mask=state_future_mask,
            )

        if self.dream_vision:
            expected = (self.vision_horizon, self.vision_target_dim)
            if (
                future_vision_target is None
                or future_vision_target.ndim != 3
                or future_vision_target.shape[1:] != expected
            ):
                shape = None if future_vision_target is None else tuple(future_vision_target.shape)
                raise ValueError(f"vision-JEPA requires [B, {expected[0]}, {expected[1]}], got {shape}")
            losses["vision_jepa_loss"] = jepa_loss(
                self.vision_dream_head(tactile_trunk),
                future_vision_target,
                beta=self.jepa_beta,
                mask=vision_future_mask,
            )
        return losses

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        tactile: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None
        if state_features is not None and state_features.ndim == 2:
            state_features = state_features.unsqueeze(1)
        tactile_features = None
        if self.use_tactile:
            if tactile is None:
                tactile = torch.zeros(
                    batch_size, self.tactile_encoder.raw_dim, dtype=vl_embs.dtype, device=device
                )
            elif tactile.ndim == 3:
                tactile = tactile[:, 0]
            tactile_features = self.tactile_encoder(tactile)

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            token_groups = []
            if state_features is not None:
                token_groups.append(state_features)
            token_groups.append(future_tokens)
            if tactile_features is not None:
                token_groups.append(tactile_features)
            token_groups.append(action_features)
            sa_embs = torch.cat(token_groups, dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @torch.no_grad()
    def sync_jepa_teachers(self) -> None:
        if not self.use_tactile_dream:
            return
        self.tactile_target_encoder.load_state_dict(self.tactile_encoder.state_dict())
        if self.dream_state:
            self.state_target_encoder.load_state_dict(self.state_encoder.state_dict())

    @torch.no_grad()
    def update_jepa_teachers(self) -> None:
        if not self.use_tactile_dream:
            return
        ema_update(self.tactile_target_encoder, self.tactile_encoder, self.ema_decay)
        if self.dream_state:
            ema_update(self.state_target_encoder, self.state_encoder, self.ema_decay)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.use_tactile_dream:
            self.tactile_target_encoder.eval()
            if self.dream_state:
                self.state_target_encoder.eval()
        return self

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(full_config=config)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
