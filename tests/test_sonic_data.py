from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.data_config import UnitreeG1SonicDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.lerobot_datasets import get_vla_dataset

DATASET_PATH = Path("/home/wzh/Projects/Uni_VLaT/data/desk_sweep")


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
        "use_tactile_temporal": True,
        "tactile_history_length": 4,
        "use_delta_targets": True,
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

    input_only = config.modality_config_for(
        _data_config(tactile_mode="input", use_tactile_temporal=False)
    )
    assert input_only["tactile"].delta_indices == [0]

    dream = config.modality_config_for(_data_config())
    assert dream["tactile"].delta_indices == list(range(-3, 5))
    assert dream["state"].delta_indices == list(range(5))
    assert dream["video"].delta_indices == list(range(5))


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="desk_sweep is not installed")
def test_train_val_split_has_no_episode_overlap():
    cfg = _data_config(
        data_root_dir=str(DATASET_PATH.parent),
        data_mix="desk_sweep",
        val_ratio=0.05,
    )
    train = get_vla_dataset(cfg, mode="train", seed=42)
    val = get_vla_dataset(cfg, mode="val", seed=42)
    train_ids = {trajectory_id for trajectory_id, _ in train.datasets[0].all_steps}
    val_ids = {trajectory_id for trajectory_id, _ in val.datasets[0].all_steps}

    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == set(train.datasets[0].trajectory_ids)


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="desk_sweep is not installed")
@pytest.mark.parametrize(
    ("mode", "expected_keys", "tactile_shape"),
    [
        ("notac", {"action", "action_mask", "image", "lang", "robot_tag", "state"}, None),
        (
            "input",
            {"action", "action_mask", "image", "lang", "robot_tag", "state", "tactile"},
            (1, 768),
        ),
    ],
)
def test_real_sonic_ablation_samples(mode, expected_keys, tactile_shape):
    data_config = UnitreeG1SonicDataConfig()
    cfg = _data_config(
        tactile_mode=mode,
        use_tactile_temporal=mode == "dream",
    )
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


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="desk_sweep is not installed")
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
    assert first["tactile"].shape == (8, 768)
    assert first["tactile"].dtype == np.uint8
    assert first["action_mask"].all()
    assert first["tactile_future_mask"].all()
    assert first["state_future_mask"].all()
    assert first["vision_future_mask"].all()
    assert len(first["image"]) == 2
    assert len(first["future_images"]) == 4
    assert all(len(frame) == 2 for frame in first["future_images"])
    assert all(
        np.array_equal(tail["tactile"][3], value) for value in tail["tactile"][4:]
    )
    assert all(np.array_equal(tail["state"][0], value) for value in tail["state"][1:])
    current_images = [np.asarray(image) for image in tail["image"]]
    for frame in tail["future_images"]:
        assert all(np.array_equal(current_images[view], np.asarray(frame[view])) for view in range(2))
    assert tail["action_mask"][0].all()
    assert not tail["action_mask"][1:].any()
    assert not tail["tactile_future_mask"].any()
    assert not tail["state_future_mask"].any()
    assert not tail["vision_future_mask"].any()
