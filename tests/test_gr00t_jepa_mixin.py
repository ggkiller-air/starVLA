from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.framework.VLM4A.gr00t_jepa import GR00TJEPAFrameworkMixin
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead


class _FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


class _FakeVisualModel(nn.Module):
    def __init__(self, return_deepstack=False):
        super().__init__()
        self.visual = _FakeVisual()
        self.return_deepstack = return_deepstack

    def get_image_features(self, pixel_values, image_grid_thw):
        features = tuple(value.reshape(1, 1).repeat(2, 4) for value in pixel_values)
        return (features, ()) if self.return_deepstack else features


class _FakeConditionalModel(nn.Module):
    def __init__(self, return_deepstack=False):
        super().__init__()
        self.model = _FakeVisualModel(return_deepstack=return_deepstack)


class _FakeInterface(nn.Module):
    def __init__(self, return_deepstack=False):
        super().__init__()
        self.model = _FakeConditionalModel(return_deepstack=return_deepstack)

    def build_qwenvl_inputs(self, images, instructions):
        values = torch.tensor([float(sample[0]) for sample in images])
        return {
            "pixel_values": values,
            "image_grid_thw": torch.ones(len(values), 3, dtype=torch.long),
        }


class _Harness(GR00TJEPAFrameworkMixin, nn.Module):
    def __init__(self, return_deepstack=False):
        super().__init__()
        self.action_model = SimpleNamespace(
            dream_vision=True,
            vision_horizon=2,
            vision_target_dim=4,
        )
        self.qwen_vl_interface = _FakeInterface(return_deepstack=return_deepstack)


class _CaptureAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.dream_state = False
        self.dream_vision = False
        self.use_tactile = False
        self.tactile_mode = "notac"
        self.dream_horizon = 2
        self.vision_horizon = 2
        self.captured_state = None

    def forward(self, hidden, actions, *, state, **kwargs):
        self.captured_state = state.detach().clone()
        return hidden.sum() * 0.0 + actions.sum() * 0.0

    def predict_action(self, hidden, *, state, **kwargs):
        self.captured_state = state.detach().clone()
        return hidden.new_zeros(hidden.shape[0], 4, 6)


class _StateHarness(GR00TJEPAFrameworkMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.action_model = _CaptureAction()
        self.action_horizon = 4
        self.config = OmegaConf.create(
            {
                "framework": {"action_model": {"repeated_diffusion_steps": 1}},
                "datasets": {"vla_data": {}},
            }
        )


def _action_config(mode="dream", dream_state=True, dream_vision=True):
    return OmegaConf.create(
        {
            "framework": {
                "action_model": {
                    "action_model_type": "DiT-B",
                    "hidden_size": 32,
                    "action_dim": 6,
                    "state_dim": 5,
                    "action_horizon": 4,
                    "num_inference_timesteps": 2,
                    "num_target_vision_tokens": 2,
                    "add_pos_embed": True,
                    "max_seq_len": 32,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 100,
                    "tactile_mode": mode,
                    "tactile_encoder_type": "mlp",
                    "n_tactile_tokens": 2,
                    "tactile_hidden_dim": 16,
                    "dream_horizon": 2,
                    "vision_horizon": 2,
                    "dream_hidden_dim": 32,
                    "dream_state": dream_state,
                    "dream_vision": dream_vision,
                    "vision_target_dim": 64,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 64,
                        "output_dim": 64,
                        "num_layers": 1,
                        "dropout": 0.0,
                        "final_dropout": False,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "positional_embeddings": None,
                    },
                }
            }
        }
    )


class _CheckpointHarness(GR00TJEPAFrameworkMixin, nn.Module):
    def __init__(self, mode="dream"):
        super().__init__()
        self.action_model = FlowmatchingActionHead(_action_config(mode=mode))


@torch.no_grad()
def test_future_vision_target_preserves_sample_time_view_order():
    harness = _Harness()
    harness._init_gr00t_jepa()
    visual = harness._visual_module()
    assert not visual.training
    assert all(not parameter.requires_grad for parameter in visual.parameters())
    harness.train()
    assert harness.training
    assert not visual.training
    assert all(not parameter.requires_grad for parameter in visual.parameters())
    examples = [
        {"future_images": [[1, 3], [5, 7]]},
        {"future_images": [[11, 13], [15, 17]]},
    ]
    target = harness._encode_future_vision_targets(examples, dtype=torch.float32)
    assert target.shape == (2, 2, 4)
    assert torch.equal(target[:, :, 0], torch.tensor([[2.0, 6.0], [12.0, 16.0]]))


@torch.no_grad()
def test_qwen3_deepstack_return_uses_primary_image_features():
    harness = _Harness(return_deepstack=True)
    target = harness._encode_future_vision_targets([{"future_images": [[2], [4]]}], dtype=torch.float32)
    assert torch.equal(target[0, :, 0], torch.tensor([2.0, 4.0]))


def test_notac_preserves_full_state_history_for_train_and_inference():
    harness = _StateHarness()
    state = np.arange(2 * 16 * 5, dtype=np.float32).reshape(2, 16, 5)
    examples = [
        {"action": np.zeros((4, 6), dtype=np.float32), "state": state[index]}
        for index in range(2)
    ]
    hidden = torch.randn(2, 3, 8)
    harness._forward_gr00t_action(examples, hidden, None)
    assert torch.equal(harness.action_model.captured_state, torch.from_numpy(state))
    harness._predict_gr00t_action(examples, hidden, None)
    assert torch.equal(harness.action_model.captured_state, torch.from_numpy(state))


def test_checkpoint_validator_accepts_only_coherent_checkpoint_generations():
    harness = _CheckpointHarness()
    all_jepa = {
        key
        for key in harness.state_dict()
        if any(key.startswith(prefix) for prefix in harness._JEPA_CHECKPOINT_PREFIXES)
    }
    dream_only = {
        key
        for key in all_jepa
        if any(key.startswith(prefix) for prefix in harness._JEPA_DREAM_PREFIXES)
    }

    harness.validate_jepa_checkpoint_keys([], [])
    assert not harness._jepa_teacher_keys_missing

    notac_result = harness.load_state_dict(_CheckpointHarness("notac").state_dict(), strict=False)
    assert set(notac_result.missing_keys) == all_jepa
    harness.validate_jepa_checkpoint_keys(notac_result.missing_keys, notac_result.unexpected_keys)
    assert harness._jepa_teacher_keys_missing

    input_result = harness.load_state_dict(_CheckpointHarness("input").state_dict(), strict=False)
    assert set(input_result.missing_keys) == dream_only
    harness.validate_jepa_checkpoint_keys(input_result.missing_keys, input_result.unexpected_keys)
    assert harness._jepa_teacher_keys_missing

    corrupted = dict(harness.state_dict())
    corrupted.pop(next(key for key in all_jepa if key.startswith("action_model.tactile_encoder.")))
    corrupted_result = harness.load_state_dict(corrupted, strict=False)
    with pytest.raises(RuntimeError, match="partial/invalid"):
        harness.validate_jepa_checkpoint_keys(
            corrupted_result.missing_keys, corrupted_result.unexpected_keys
        )
    with pytest.raises(RuntimeError, match="unexpected"):
        harness.validate_jepa_checkpoint_keys([], ["old_head.weight"])
