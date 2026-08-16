"""Shared JEPA data plumbing for Qwen/Cosmos GR00T frameworks."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class GR00TJEPAFrameworkMixin:
    _JEPA_CHECKPOINT_PREFIXES = (
        "action_model.tactile_encoder.",
        "action_model.tactile_temporal_encoder.",
        "action_model.tactile_target_encoder.",
        "action_model.tactile_dream_head.",
        "action_model.state_target_encoder.",
        "action_model.state_dream_head.",
        "action_model.vision_dream_head.",
    )
    _JEPA_TEACHER_PREFIXES = (
        "action_model.tactile_target_encoder.",
        "action_model.state_target_encoder.",
    )
    _JEPA_DREAM_PREFIXES = (
        "action_model.tactile_target_encoder.",
        "action_model.tactile_dream_head.",
        "action_model.state_target_encoder.",
        "action_model.state_dream_head.",
        "action_model.vision_dream_head.",
    )

    def _init_gr00t_jepa(self) -> None:
        if not self.action_model.dream_vision:
            return
        visual = self._visual_module()
        visual.requires_grad_(False)
        visual.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.action_model.dream_vision:
            visual = self._visual_module()
            visual.requires_grad_(False)
            visual.eval()
        return self

    def _visual_model(self):
        model = self.qwen_vl_interface.model
        base = getattr(model, "model", None)
        if base is None or not hasattr(base, "get_image_features"):
            raise RuntimeError("vision-JEPA requires a Qwen-compatible model.model.get_image_features API")
        return base

    def _visual_module(self):
        visual = getattr(self._visual_model(), "visual", None)
        if visual is None:
            raise RuntimeError("vision-JEPA could not locate the frozen visual tower")
        return visual

    def _validate_dataset_jepa_config(self) -> None:
        data_config = getattr(getattr(self.config, "datasets", None), "vla_data", None)
        if data_config is None or data_config.get("tactile_mode", None) is None:
            return
        expected = {
            "tactile_mode": self.action_model.tactile_mode,
            "dream_horizon": self.action_model.dream_horizon,
            "dream_state": self.action_model.dream_state,
            "dream_vision": self.action_model.dream_vision,
            "vision_horizon": self.action_model.vision_horizon,
            "use_tactile_temporal": self.action_model.use_tactile_temporal,
            "tactile_history_length": self.action_model.tactile_history_length,
            "use_delta_targets": self.action_model.use_delta_targets,
        }
        for key, model_value in expected.items():
            data_value = data_config.get(key, model_value)
            if isinstance(model_value, bool):
                data_value = bool(data_value)
            elif isinstance(model_value, int):
                data_value = int(data_value)
            else:
                data_value = str(data_value).lower()
            if data_value != model_value:
                raise ValueError(f"Dataset/model JEPA config mismatch for {key}: {data_value!r} != {model_value!r}")

    @torch.no_grad()
    def _encode_future_vision_targets(
        self,
        examples: list[dict[str, Any]],
        dtype: torch.dtype,
        expected_horizon: int | None = None,
    ) -> torch.Tensor:
        expected_horizon = expected_horizon or self.action_model.vision_horizon
        windows = []
        num_views = None
        for example in examples:
            future = example.get("future_images")
            if future is None or len(future) != expected_horizon:
                length = None if future is None else len(future)
                raise ValueError(f"vision-JEPA requires {expected_horizon} future frames, got {length}")
            view_counts = {len(frame) for frame in future}
            if len(view_counts) != 1 or 0 in view_counts:
                raise ValueError(f"Every future frame must contain the same non-zero views: {view_counts}")
            current_views = view_counts.pop()
            if num_views is None:
                num_views = current_views
            elif num_views != current_views:
                raise ValueError("All samples must have the same number of future camera views")
            windows.append(future)

        flat_images = [
            [windows[batch][time][view]]
            for batch in range(len(windows))
            for time in range(expected_horizon)
            for view in range(num_views)
        ]
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=flat_images,
            instructions=[""] * len(flat_images),
        )
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is None or image_grid_thw is None:
            raise RuntimeError("Vision processor did not return pixel_values and image_grid_thw")

        visual = self._visual_module()
        visual.eval()
        image_features = self._visual_model().get_image_features(pixel_values, image_grid_thw)
        if isinstance(image_features, tuple) and len(image_features) == 2:
            image_features = image_features[0]
        if len(image_features) != len(flat_images):
            raise RuntimeError(f"Expected {len(flat_images)} visual feature groups, got {len(image_features)}")
        pooled = torch.stack([features.mean(dim=0) for features in image_features])
        pooled = pooled.reshape(len(windows), expected_horizon, num_views, pooled.shape[-1]).mean(dim=2)
        expected_dim = self.action_model.vision_target_dim
        if pooled.shape[-1] != expected_dim:
            raise ValueError(f"Visual target width {pooled.shape[-1]} does not match vision_target_dim {expected_dim}")
        return pooled.to(dtype=dtype).detach()

    @staticmethod
    def _tensorize_optional(examples, key: str, device, dtype):
        if key not in examples[0]:
            if any(key in example for example in examples[1:]):
                raise ValueError(f"Inconsistent optional key {key!r} within a batch")
            return None
        if any(key not in example for example in examples):
            raise ValueError(f"Inconsistent optional key {key!r} within a batch")
        return torch.as_tensor(np.asarray([example[key] for example in examples]), device=device, dtype=dtype)

    def _forward_gr00t_action(self, examples, last_hidden, encoder_attention_mask):
        self._validate_dataset_jepa_config()
        device, dtype = last_hidden.device, last_hidden.dtype
        actions = torch.as_tensor(np.asarray([example["action"] for example in examples]), device=device, dtype=dtype)
        actions = actions[:, -self.action_horizon :]
        action_mask = self._tensorize_optional(examples, "action_mask", device, dtype)
        if action_mask is not None:
            action_mask = action_mask[:, -self.action_horizon :]
        repeats = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))

        state_window = self._tensorize_optional(examples, "state", device, dtype)
        current_state = None
        future_state = None
        if state_window is not None:
            if state_window.ndim == 2:
                current_state = state_window.unsqueeze(1)
            elif state_window.ndim == 3:
                current_state = state_window[:, :1] if self.action_model.dream_state else state_window
            else:
                raise ValueError(f"Expected state [B, D] or [B, T, D], got {state_window.shape}")
            if self.action_model.dream_state:
                expected = self.action_model.dream_horizon + 1
                if state_window.ndim != 3 or state_window.shape[1] != expected:
                    raise ValueError(f"state-JEPA requires [B, {expected}, D], got {state_window.shape}")
                future_state = state_window[:, 1:]

        tactile = self._tensorize_optional(examples, "tactile", device, dtype)
        if self.action_model.use_tactile and tactile is None:
            raise ValueError("Active tactile mode requires the tactile key in every training sample")
        tactile_future_mask = self._tensorize_optional(
            examples, "tactile_future_mask", device, dtype
        )
        state_future_mask = self._tensorize_optional(
            examples, "state_future_mask", device, dtype
        )
        vision_future_mask = self._tensorize_optional(
            examples, "vision_future_mask", device, dtype
        )

        future_vision = None
        current_vision = None
        if self.action_model.dream_vision:
            future_vision = self._encode_future_vision_targets(examples, dtype=dtype)
            if self.action_model.use_delta_targets:
                current_vision = self._encode_future_vision_targets(
                    [{"future_images": [example["image"]]} for example in examples],
                    dtype=dtype,
                    expected_horizon=1,
                )[:, 0]

        def repeat(value):
            return None if value is None else value.repeat(repeats, *([1] * (value.ndim - 1)))

        repeated_mask = repeat(encoder_attention_mask)
        if repeated_mask is not None:
            repeated_mask = repeated_mask.to(dtype=torch.bool)
        output = self.action_model(
            repeat(last_hidden),
            repeat(actions),
            state=repeat(current_state),
            tactile=repeat(tactile),
            future_state=repeat(future_state),
            future_vision_target=repeat(future_vision),
            current_vision_target=repeat(current_vision),
            action_mask=repeat(action_mask),
            tactile_future_mask=repeat(tactile_future_mask),
            state_future_mask=repeat(state_future_mask),
            vision_future_mask=repeat(vision_future_mask),
            encoder_attention_mask=repeated_mask,
        )
        return {"action_loss": output} if torch.is_tensor(output) else output

    def _predict_gr00t_action(self, examples, last_hidden, encoder_attention_mask):
        device, dtype = last_hidden.device, last_hidden.dtype
        state = self._tensorize_optional(examples, "state", device, dtype)
        if state is not None:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            elif state.ndim == 3:
                if self.action_model.dream_state:
                    state = state[:, :1]
            else:
                raise ValueError(f"Expected state [B, D] or [B, T, D], got {state.shape}")
        tactile = self._tensorize_optional(examples, "tactile", device, dtype)
        return self.action_model.predict_action(
            last_hidden,
            state=state,
            tactile=tactile,
            encoder_attention_mask=encoder_attention_mask,
        )

    def sync_jepa_teachers(self) -> None:
        self.action_model.sync_jepa_teachers()

    def update_jepa_teachers(self) -> None:
        self.action_model.update_jepa_teachers()

    def jepa_loss_weights(self) -> dict[str, float]:
        return {
            "tactile_loss": self.action_model.lambda_tactile,
            "state_jepa_loss": self.action_model.lambda_state,
            "vision_jepa_loss": self.action_model.lambda_vision,
        }

    def validate_jepa_checkpoint_keys(self, missing_keys, unexpected_keys) -> None:
        missing = set(missing_keys)
        all_jepa = {
            key
            for key in self.state_dict()
            if any(key.startswith(prefix) for prefix in self._JEPA_CHECKPOINT_PREFIXES)
        }
        dream_only = {
            key
            for key in all_jepa
            if any(key.startswith(prefix) for prefix in self._JEPA_DREAM_PREFIXES)
        }
        valid_missing_sets = {frozenset(), frozenset(all_jepa), frozenset(dream_only)}
        if frozenset(missing) not in valid_missing_sets or unexpected_keys:
            raise RuntimeError(
                "Checkpoint is incompatible with this JEPA model: "
                f"partial/invalid missing={sorted(missing)}, unexpected={list(unexpected_keys)}"
            )
        self._jepa_teacher_keys_missing = bool(missing)
