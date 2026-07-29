from types import SimpleNamespace

import numpy as np
import pytest

from deployment.model_server.sonic_policy import (
    ACTION_KEYS,
    SonicPolicyAdapter,
    validate_actions,
    validate_observation,
)


def observation():
    return {
        "state": np.zeros(46, dtype=np.float32),
        "ego_view_left": np.zeros((8, 10, 3), dtype=np.uint8),
        "ego_view_right": np.zeros((8, 10, 3), dtype=np.uint8),
        "prompt": "carry the bucket",
        "tactile": np.zeros(256, dtype=np.uint8),
    }


def test_sonic_observation_contract_accepts_canonical_request():
    result = validate_observation(observation(), requires_tactile=True)
    assert result["state"].shape == (46,)
    assert result["tactile"].shape == (256,)


def test_sonic_observation_contract_requires_tactile_for_jepa_checkpoint():
    obs = observation()
    del obs["tactile"]
    with pytest.raises(ValueError, match="requires tactile"):
        validate_observation(obs, requires_tactile=True)


def test_sonic_action_contract_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(ValueError, match="shape"):
        validate_actions(np.zeros((40, 32), dtype=np.float32))
    actions = np.zeros((40, 78), dtype=np.float32)
    actions[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or infinity"):
        validate_actions(actions)


class FakeStarPolicy:
    def __init__(self):
        self._framework = SimpleNamespace(
            action_model=SimpleNamespace(use_tactile=True)
        )
        self._model_cfg = {
            "framework": {
                "action_model": {
                    "action_dim": 78,
                    "state_dim": 46,
                }
            }
        }
        self._action_chunk_size = 40
        self.example = None

    def _get_processor(self, _unnorm_key):
        return SimpleNamespace(action_keys=ACTION_KEYS)

    def normalize_state(self, state, _unnorm_key):
        return state + 1.0

    def predict_action(self, *, examples, unnorm_key):
        assert unnorm_key == "carry_bucket"
        self.example = examples[0]
        return {"actions": np.zeros((1, 40, 78), dtype=np.float32)}


def test_sonic_adapter_forwards_stereo_state_and_tactile_to_starvla():
    policy = FakeStarPolicy()
    adapter = SonicPolicyAdapter(policy, unnorm_key="carry_bucket")
    obs = observation()
    obs["ego_view_left"].fill(1)
    obs["ego_view_right"].fill(2)

    result = adapter.infer(obs)

    assert result["actions"].shape == (40, 78)
    assert policy.example is not None
    assert len(policy.example["image"]) == 2
    assert np.all(policy.example["image"][0] == 1)
    assert np.all(policy.example["image"][1] == 2)
    np.testing.assert_array_equal(policy.example["state"], np.ones(46, dtype=np.float32))
    np.testing.assert_array_equal(policy.example["tactile"], obs["tactile"])
