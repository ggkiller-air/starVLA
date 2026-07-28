from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

from starVLA.training import train_starvla


def test_build_accelerator_wires_yaml_gradient_accumulation(monkeypatch):
    captured = {}

    class _FakeAccelerator:
        state = "fake-state"

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def print(self, value):
            captured["printed"] = value

    monkeypatch.setattr(train_starvla, "Accelerator", _FakeAccelerator)
    monkeypatch.setattr(train_starvla, "DeepSpeedPlugin", lambda: "plugin")
    cfg = SimpleNamespace(trainer={"gradient_accumulation_steps": 4})

    accelerator = train_starvla.build_accelerator(cfg)

    assert isinstance(accelerator, _FakeAccelerator)
    assert captured["deepspeed_plugin"] == "plugin"
    assert captured["gradient_accumulation_steps"] == 4
    assert captured["printed"] == "fake-state"


class _LossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.teacher_updates = 0

    def forward(self, _batch):
        loss = self.weight.square()
        return {"action_loss": loss, "tactile_loss": loss}

    @staticmethod
    def jepa_loss_weights():
        return {"tactile_loss": 0.5}

    def update_jepa_teachers(self):
        self.teacher_updates += 1


class _AccumulatingOptimizer:
    def __init__(self, optimizer, accelerator):
        self.optimizer = optimizer
        self.accelerator = accelerator

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        if self.accelerator.sync_gradients:
            self.optimizer.step()

    def zero_grad(self):
        if self.accelerator.sync_gradients:
            self.optimizer.zero_grad()


class _FakeAccumulatingAccelerator:
    num_processes = 1
    gradient_accumulation_steps = 2
    optimizer_step_was_skipped = False

    def __init__(self):
        self.sync_gradients = False

    @staticmethod
    def accumulate(_model):
        return nullcontext()

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def backward(loss):
        loss.backward()


class _CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_train_step_accumulates_weighted_loss_and_updates_teacher_once():
    model = _LossModel()
    accelerator = _FakeAccumulatingAccelerator()
    optimizer = _AccumulatingOptimizer(torch.optim.SGD(model.parameters(), lr=0.1), accelerator)
    scheduler = _CountingScheduler()
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(gradient_clipping=None),
        datasets=SimpleNamespace(vla_data=SimpleNamespace(per_device_batch_size=1)),
    )
    trainer = train_starvla.VLATrainer(cfg, model, [], optimizer, scheduler, accelerator)

    first_metrics = trainer._train_step(None)
    assert first_metrics["total_loss"] == 1.5
    assert model.weight.item() == 1.0
    assert model.teacher_updates == 0
    assert scheduler.steps == 0

    accelerator.sync_gradients = True
    trainer._train_step(None)
    assert torch.isclose(model.weight, torch.tensor(0.4))
    assert model.teacher_updates == 1
    assert scheduler.steps == 1
    assert model.weight.grad is None
