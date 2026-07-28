from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.framework.VLM4A.gr00t_jepa import GR00TJEPAFrameworkMixin


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


def test_checkpoint_validator_only_allows_new_jepa_modules():
    harness = _Harness()
    harness.validate_jepa_checkpoint_keys(["action_model.tactile_encoder.per_region.weight"], [])
    assert not harness._jepa_teacher_keys_missing
    harness.validate_jepa_checkpoint_keys(["action_model.tactile_target_encoder.per_region.weight"], [])
    assert harness._jepa_teacher_keys_missing
    with pytest.raises(RuntimeError, match="invalid missing"):
        harness.validate_jepa_checkpoint_keys(["action_model.model.weight"], [])
    with pytest.raises(RuntimeError, match="unexpected"):
        harness.validate_jepa_checkpoint_keys([], ["old_head.weight"])
