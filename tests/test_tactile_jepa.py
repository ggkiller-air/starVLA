from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.action_model.tactile_jepa import (
    RAW_DIM,
    REGION_GRIDS,
    REGION_SIZES,
    VALID_IDX,
    TactileEncoder,
    TactileTemporalEncoder,
    build_ema_teacher,
    ema_update,
    jepa_loss,
)


def _action_config(mode="notac", dream_state=False, dream_vision=False):
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
                    "use_tactile_temporal": mode == "dream" and dream_state,
                    "tactile_history_length": 4,
                    "use_delta_targets": mode == "dream" and dream_state,
                    "vision_target_dim": 64,
                    "ema_decay": 0.9,
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


def test_sonic_layout_and_single_normalization():
    assert len(VALID_IDX) == sum(REGION_SIZES) == 624
    assert len(set(VALID_IDX)) == len(VALID_IDX)
    assert min(VALID_IDX) >= 0 and max(VALID_IDX) < RAW_DIM
    assert tuple(rows * cols for rows, cols in REGION_GRIDS) == REGION_SIZES
    encoder = TactileEncoder(embed_dim=16, hidden_dim=8, num_tokens=2, num_heads=2)
    raw = torch.stack((torch.zeros(RAW_DIM), torch.full((RAW_DIM,), 255.0)))
    normalized = encoder.select_and_normalize(raw)
    assert torch.equal(normalized[0], torch.zeros_like(normalized[0]))
    assert torch.equal(normalized[1], torch.ones_like(normalized[1]))


@pytest.mark.parametrize("encoder_type", ["mlp", "cnn", "coord"])
def test_tactile_encoder_variants_shape_and_backward(encoder_type):
    encoder = TactileEncoder(
        embed_dim=32,
        hidden_dim=16,
        num_tokens=4,
        num_heads=4,
        encoder_type=encoder_type,
    )
    output = encoder(torch.randint(0, 256, (2, RAW_DIM), dtype=torch.uint8))
    assert output.shape == (2, 4, 32)
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())


def test_ema_teacher_is_frozen_and_moves_toward_student():
    student = TactileEncoder(embed_dim=16, hidden_dim=8, num_tokens=2, num_heads=2)
    teacher = build_ema_teacher(student)
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    with torch.no_grad():
        next(student.parameters()).add_(1.0)
    before = next(teacher.parameters()).clone()
    ema_update(teacher, student, decay=0.5)
    after = next(teacher.parameters())
    assert not torch.equal(before, after)
    assert torch.allclose(after, 0.5 * before + 0.5 * next(student.parameters()))


def test_temporal_encoder_shape_and_zero_delta_loss():
    encoder = TactileTemporalEncoder(16, 4, num_heads=4)
    assert encoder(torch.randn(2, 4, 3, 16)).shape == (2, 3, 16)
    zero = torch.zeros(2, 4, 16)
    assert jepa_loss(zero, zero) == 0


def test_notac_keeps_baseline_graph_and_scalar_loss():
    model = FlowmatchingActionHead(_action_config())
    assert not hasattr(model, "tactile_encoder")
    loss = model(torch.randn(2, 7, 64), torch.randn(2, 4, 6), torch.randn(2, 1, 5))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_input_mode_backpropagates_action_loss_without_teacher():
    model = FlowmatchingActionHead(_action_config(mode="input"))
    loss = model(
        torch.randn(2, 7, 64),
        torch.randn(2, 4, 6),
        torch.randn(2, 1, 5),
        tactile=torch.randint(0, 256, (2, 1, RAW_DIM), dtype=torch.uint8),
    )
    loss.backward()
    assert not hasattr(model, "tactile_target_encoder")
    assert any(parameter.grad is not None for parameter in model.tactile_encoder.parameters())


def test_dream_mode_returns_all_losses_and_never_grads_teachers():
    model = FlowmatchingActionHead(_action_config(mode="dream", dream_state=True, dream_vision=True))
    output = model(
        torch.randn(2, 7, 64),
        torch.randn(2, 4, 6),
        torch.randn(2, 1, 5),
        tactile=torch.randint(0, 256, (2, 6, RAW_DIM), dtype=torch.uint8),
        future_state=torch.randn(2, 2, 5),
        future_vision_target=torch.randn(2, 2, 64),
        current_vision_target=torch.randn(2, 64),
    )
    assert set(output) == {"action_loss", "tactile_loss", "state_jepa_loss", "vision_jepa_loss"}
    assert all(torch.isfinite(loss) for loss in output.values())
    sum(output.values()).backward()
    teachers = (model.tactile_target_encoder, model.state_target_encoder)
    assert all(parameter.grad is None for teacher in teachers for parameter in teacher.parameters())
    assert any(parameter.grad is not None for parameter in model.tactile_dream_head.parameters())


def test_htd_returns_only_action_and_future_tactile_losses():
    model = FlowmatchingActionHead(_action_config(mode="dream"))
    output = model(
        torch.randn(2, 7, 64),
        torch.randn(2, 4, 6),
        torch.randn(2, 1, 5),
        tactile=torch.randint(0, 256, (2, 3, RAW_DIM), dtype=torch.uint8),
    )
    assert set(output) == {"action_loss", "tactile_loss"}


def test_time_masks_remove_episode_padding_from_losses():
    prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    first_target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    second_target = first_target.clone()
    second_target[:, 1] = torch.tensor([-10.0, 7.0])
    future_mask = torch.tensor([[True, False]])
    assert torch.equal(
        jepa_loss(prediction, first_target, mask=future_mask),
        jepa_loss(prediction, second_target, mask=future_mask),
    )

    model = FlowmatchingActionHead(_action_config(mode="notac"))
    vl_embs = torch.randn(2, 7, 64)
    state = torch.randn(2, 1, 5)
    first_actions = torch.randn(2, 4, 6)
    second_actions = first_actions.clone()
    second_actions[:, -1] = 1000
    action_mask = torch.ones_like(first_actions)
    action_mask[:, -1] = 0
    torch.manual_seed(456)
    first_loss = model(
        vl_embs,
        first_actions,
        state,
        action_mask=action_mask,
    )
    torch.manual_seed(456)
    second_loss = model(
        vl_embs,
        second_actions,
        state,
        action_mask=action_mask,
    )
    assert torch.equal(first_loss, second_loss)


def test_future_targets_do_not_condition_action_prediction():
    model = FlowmatchingActionHead(_action_config(mode="dream", dream_state=True, dream_vision=True))
    model.eval()
    vl_embs = torch.randn(2, 7, 64)
    actions = torch.randn(2, 4, 6)
    current_state = torch.randn(2, 1, 5)
    current_tactile = torch.randint(0, 256, (2, 4, RAW_DIM), dtype=torch.uint8)

    def forward_with_future(fill):
        tactile = torch.cat(
            (current_tactile, torch.full((2, 2, RAW_DIM), fill, dtype=torch.uint8)), dim=1
        )
        torch.manual_seed(123)
        return model(
            vl_embs,
            actions,
            current_state,
            tactile=tactile,
            future_state=torch.full((2, 2, 5), float(fill)),
            future_vision_target=torch.full((2, 2, 64), float(fill)),
            current_vision_target=torch.zeros(2, 64),
        )

    first = forward_with_future(0)
    second = forward_with_future(255)
    assert torch.equal(first["action_loss"], second["action_loss"])


def test_dream_mode_fails_fast_without_future_targets():
    model = FlowmatchingActionHead(_action_config(mode="dream", dream_state=True))
    with pytest.raises(ValueError, match="state-JEPA"):
        model(
            torch.randn(1, 7, 64),
            torch.randn(1, 4, 6),
            torch.randn(1, 1, 5),
            tactile=torch.randint(0, 256, (1, 6, RAW_DIM), dtype=torch.uint8),
        )


def test_dream_inference_is_current_only_and_zero_fills_missing_tactile():
    model = FlowmatchingActionHead(_action_config(mode="dream"))
    actions = model.predict_action(torch.randn(1, 7, 64), state=torch.randn(1, 1, 5))
    assert actions.shape == (1, 4, 6)
