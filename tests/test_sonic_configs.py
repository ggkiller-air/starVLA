from pathlib import Path

import pytest
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.data_config import UnitreeG1SonicDataConfig

CONFIG_DIR = Path("examples/Sonic/train_files")
MODE_CONFIGS = {
    "notactile": ("starvla_train_sonic_notactile.yaml", "notac", False, False),
    "htd": ("starvla_train_sonic_htd.yaml", "dream", False, False),
    "jepa": ("starvla_train_sonic_jepa.yaml", "dream", True, True),
}


@pytest.mark.parametrize(("mode", "expected"), MODE_CONFIGS.items())
def test_table1_configs_have_fixed_sonic_contract_and_matching_data(mode, expected):
    filename, tactile_mode, dream_state, dream_vision = expected
    config = OmegaConf.load(CONFIG_DIR / filename)
    model = config.framework.action_model
    data = config.datasets.vla_data

    assert config.run_id == f"sonic_{mode}"
    assert (model.action_horizon, model.action_dim, model.state_dim) == (40, 78, 46)
    assert model.tactile_mode == data.tactile_mode == tactile_mode
    assert bool(model.dream_state) is bool(data.dream_state) is dream_state
    assert bool(model.dream_vision) is bool(data.dream_vision) is dream_vision

    modalities = UnitreeG1SonicDataConfig().modality_config_for(data)
    assert modalities["video"].modality_keys == [
        "video.ego_view_left",
        "video.ego_view_right",
    ]
    assert sum(UnitreeG1SonicDataConfig.state_key_dims.values()) == 46
    assert sum(UnitreeG1SonicDataConfig.action_key_dims.values()) == 78
    assert UnitreeG1SonicDataConfig.action_indices == list(range(40))

    if mode == "notactile":
        assert "tactile" not in modalities
        assert modalities["state"].delta_indices == [0]
        assert modalities["video"].delta_indices == [0]
    elif mode == "htd":
        assert modalities["tactile"].delta_indices == list(range(5))
        assert modalities["state"].delta_indices == [0]
        assert modalities["video"].delta_indices == [0]
    else:
        assert modalities["tactile"].delta_indices == list(range(5))
        assert modalities["state"].delta_indices == list(range(5))
        assert modalities["video"].delta_indices == list(range(5))
