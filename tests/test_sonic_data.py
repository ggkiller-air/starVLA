from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.data_config import UnitreeG1SonicDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset

DATASET_PATH = Path("/root/Projects/data/carry-bucket-stereo")


def _data_config(**overrides):
    values = {
        "lerobot_version": "v2.0",
        "action_mode": "abs",
        "include_state": True,
        "tactile_mode": "dream",
        "dream_horizon": 4,
        "dream_state": True,
        "dream_vision": True,
        "vision_horizon": 4,
        "video_backend": "torchvision_av",
    }
    values.update(overrides)
    return OmegaConf.create(values)


def test_sonic_modalities_follow_ablation_mode():
    config = UnitreeG1SonicDataConfig()
    notac = config.modality_config_for(_data_config(tactile_mode="notac"))
    assert "tactile" not in notac
    assert notac["state"].delta_indices == [0]
    assert notac["video"].delta_indices == [0]

    input_only = config.modality_config_for(_data_config(tactile_mode="input"))
    assert input_only["tactile"].delta_indices == [0]

    dream = config.modality_config_for(_data_config())
    assert dream["tactile"].delta_indices == list(range(5))
    assert dream["state"].delta_indices == list(range(5))
    assert dream["video"].delta_indices == list(range(5))


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="carry-bucket-stereo is not installed")
@pytest.mark.parametrize(
    ("mode", "expected_keys", "tactile_shape"),
    [
        ("notac", {"action", "action_mask", "image", "lang", "robot_tag", "state"}, None),
        (
            "input",
            {"action", "action_mask", "image", "lang", "robot_tag", "state", "tactile"},
            (1, 256),
        ),
    ],
)
def test_real_sonic_ablation_samples(mode, expected_keys, tactile_shape):
    data_config = UnitreeG1SonicDataConfig()
    cfg = _data_config(tactile_mode=mode)
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=data_config.modality_config_for(cfg),
        embodiment_tag=data_config.embodiment_tag,
        transforms=data_config.transform(),
        video_backend="torchvision_av",
        data_cfg=cfg,
    )
    sample = dataset[0]
    assert set(sample) == expected_keys
    assert sample["state"].shape == (1, 46)
    if tactile_shape is not None:
        assert sample["tactile"].shape == tactile_shape


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="carry-bucket-stereo is not installed")
def test_real_sonic_sample_and_episode_tail_padding():
    data_config = UnitreeG1SonicDataConfig()
    cfg = _data_config()
    dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=data_config.modality_config_for(cfg),
        embodiment_tag=data_config.embodiment_tag,
        transforms=data_config.transform(),
        video_backend="torchvision_av",
        data_cfg=cfg,
    )
    first = dataset[0]
    tail = dataset[int(dataset.trajectory_lengths[0]) - 1]
    assert first["action"].shape == (40, 78)
    assert first["state"].shape == (5, 46)
    assert first["tactile"].shape == (5, 256)
    assert first["tactile"].dtype == np.uint8
    assert first["action_mask"].all()
    assert first["tactile_future_mask"].all()
    assert first["state_future_mask"].all()
    assert first["vision_future_mask"].all()
    assert len(first["image"]) == 2
    assert len(first["future_images"]) == 4
    assert all(len(frame) == 2 for frame in first["future_images"])
    assert all(np.array_equal(tail["tactile"][0], value) for value in tail["tactile"][1:])
    assert all(np.array_equal(tail["state"][0], value) for value in tail["state"][1:])
    current_images = [np.asarray(image) for image in tail["image"]]
    for frame in tail["future_images"]:
        assert all(np.array_equal(current_images[view], np.asarray(frame[view])) for view in range(2))
    assert tail["action_mask"][0].all()
    assert not tail["action_mask"][1:].any()
    assert not tail["tactile_future_mask"].any()
    assert not tail["state_future_mask"].any()
    assert not tail["vision_future_mask"].any()
