"""Canonical SONIC websocket adapter for starVLA checkpoints."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

PROTOCOL = "sonic_vla_v1"
STATE_DIM = 46
ACTION_HORIZON = 40
ACTION_DIM = 78
TACTILE_DIM = 768
VIDEO_KEYS = ("ego_view_left", "ego_view_right")
ACTION_KEYS = (
    "action.motion_token",
    "action.left_hand_joints",
    "action.right_hand_joints",
)


def validate_observation(
    observation: Mapping[str, Any], *, requires_tactile: bool, tactile_history_length: int = 1
) -> dict:
    state = np.asarray(observation.get("state"))
    if state.dtype != np.float32 or state.shape != (STATE_DIM,):
        raise ValueError(f"state must be float32[{STATE_DIM}], got {state.dtype} {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("state contains NaN or infinity")

    result = {"state": state}
    image_shape = None
    for key in VIDEO_KEYS:
        image = np.asarray(observation.get(key))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be uint8[H, W, 3], got {image.dtype} {image.shape}")
        image_shape = image.shape if image_shape is None else image_shape
        if image.shape != image_shape:
            raise ValueError("Stereo images must have identical shapes")
        result[key] = image

    prompt = observation.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    result["prompt"] = prompt

    tactile_value = observation.get("tactile")
    if requires_tactile and tactile_value is None:
        raise ValueError(f"This starVLA checkpoint requires tactile uint8[{TACTILE_DIM}]")
    if tactile_value is not None:
        tactile = np.asarray(tactile_value)
        expected_shape = (
            (TACTILE_DIM,) if tactile_history_length == 1 else (tactile_history_length, TACTILE_DIM)
        )
        if tactile.dtype != np.uint8 or tactile.shape != expected_shape:
            raise ValueError(
                f"tactile must have shape {expected_shape} and dtype uint8, "
                f"got {tactile.dtype} {tactile.shape}"
            )
        result["tactile"] = tactile
    return result


def validate_actions(actions: Any) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"actions must have shape ({ACTION_HORIZON}, {ACTION_DIM}), got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or infinity")
    return actions


class SonicPolicyAdapter:
    """Translate the flat SONIC wire request into a starVLA example."""

    def __init__(self, policy, *, unnorm_key: str | None = None) -> None:
        self._policy = policy
        self._unnorm_key = unnorm_key
        action_model = policy._framework.action_model
        self.requires_tactile = bool(getattr(action_model, "use_tactile", False))
        self.tactile_history_length = (
            int(getattr(action_model, "tactile_history_length", 1))
            if self.requires_tactile
            else 0
        )

        action_cfg = policy._model_cfg["framework"]["action_model"]
        if int(action_cfg["action_dim"]) != ACTION_DIM:
            raise ValueError(f"SONIC checkpoint action_dim must be {ACTION_DIM}")
        if int(action_cfg["state_dim"]) != STATE_DIM:
            raise ValueError(f"SONIC checkpoint state_dim must be {STATE_DIM}")
        if policy._action_chunk_size != ACTION_HORIZON:
            raise ValueError(f"SONIC checkpoint action horizon must be {ACTION_HORIZON}")

        processor = policy._get_processor(unnorm_key)
        if tuple(processor.action_keys) != ACTION_KEYS:
            raise ValueError(
                f"SONIC action key order must be {ACTION_KEYS}, got {processor.action_keys}"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "backend": "starvla",
            "state_dim": STATE_DIM,
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "video_keys": list(VIDEO_KEYS),
            "requires_tactile": self.requires_tactile,
            "tactile_history_length": self.tactile_history_length,
            "action_layout": {
                "motion_token": [0, 64],
                "left_hand_joints": [64, 71],
                "right_hand_joints": [71, 78],
            },
        }

    def infer(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        obs = validate_observation(
            observation,
            requires_tactile=self.requires_tactile,
            tactile_history_length=self.tactile_history_length,
        )
        example = {
            "image": [obs["ego_view_left"], obs["ego_view_right"]],
            "lang": obs["prompt"],
            "state": self._policy.normalize_state(obs["state"], self._unnorm_key),
        }
        if "tactile" in obs:
            example["tactile"] = obs["tactile"]

        output = self._policy.predict_action(
            examples=[example],
            unnorm_key=self._unnorm_key,
        )
        batched_actions = np.asarray(output["actions"])
        if batched_actions.shape != (1, ACTION_HORIZON, ACTION_DIM):
            raise ValueError(
                "starVLA policy must return batched actions with shape "
                f"(1, {ACTION_HORIZON}, {ACTION_DIM}), got {batched_actions.shape}"
            )
        return {"actions": validate_actions(batched_actions[0])}
