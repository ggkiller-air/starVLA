# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import (
    TrainerUtils,
    build_param_lr_groups,
    normalize_dotlist_args,
)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)

COMPARISON_GROUP = "sonic-htd-model-comparison"
DEFAULT_CHECKPOINT_COUNT = 5


def comparison_metrics(metrics, step, learning_rates):
    """Return the common W&B schema used across all five VLA trainers."""
    aliases = {
        "comparison/step": step,
        "comparison/loss": metrics.get("total_loss"),
        "comparison/action_loss": metrics.get("action_dit_loss"),
        "comparison/tactile_loss": metrics.get("tactile_loss"),
        "comparison/vision_loss": metrics.get("vision_jepa_loss"),
        "val/action_mse": metrics.get("val_action_mse"),
    }
    if learning_rates:
        aliases["comparison/lr"] = learning_rates[-1]
    return {key: value for key, value in aliases.items() if value is not None}


def checkpoint_milestones(max_train_steps, checkpoint_count=DEFAULT_CHECKPOINT_COUNT):
    """Return evenly spaced checkpoint steps, including the final step."""
    max_train_steps = int(max_train_steps)
    checkpoint_count = int(checkpoint_count)
    if max_train_steps <= 0:
        raise ValueError("max_train_steps must be positive")
    if checkpoint_count <= 0:
        raise ValueError("checkpoint_count must be positive")
    if max_train_steps % checkpoint_count != 0:
        raise ValueError(
            f"max_train_steps ({max_train_steps}) must be divisible by "
            f"checkpoint_count ({checkpoint_count})"
        )
    interval = max_train_steps // checkpoint_count
    return tuple(interval * index for index in range(1, checkpoint_count + 1))


def build_accelerator(cfg):
    gradient_accumulation_steps = int(cfg.trainer.get("gradient_accumulation_steps", 1))
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(),
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    accelerator.print(accelerator.state)
    return accelerator


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> tuple[DataLoader, DataLoader]:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(
        cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, mode="train"
    )
    vla_val_dataloader = build_dataloader(
        cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py, mode="val"
    )

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader, vla_val_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(
        self,
        cfg,
        model,
        vla_train_dataloader,
        optimizer,
        lr_scheduler,
        accelerator,
        vla_val_dataloader=None,
    ):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vla_val_dataloader = vla_val_dataloader or vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        if getattr(self, "_sync_jepa_teachers_after_load", False) and hasattr(
            self.model, "sync_jepa_teachers"
        ):
            self.model.sync_jepa_teachers()
            logger.info("Synchronized JEPA teachers from checkpoint-loaded online encoders")
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        (
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.vla_val_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.vla_val_dataloader,
            self.lr_scheduler,
        )

        if self._resume_state_checkpoint is not None:
            self._load_checkpoint(self._resume_state_checkpoint)

        if self.completed_steps == 0 and self._resume_state_checkpoint is None:
            logger.info("Saving step-0 smoke checkpoint before training")
            self._save_checkpoint()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases."""
        if not bool(self.config.get("wandb_enabled", True)):
            if self.accelerator.is_main_process:
                wandb.init(mode="disabled")
            return
        if self.accelerator.is_main_process:
            wandb.init(
                name=f"StarVLA / {self.config.run_id}",
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group=COMPARISON_GROUP,
                job_type="comparison-training",
            )
            wandb.define_metric("comparison/*", step_metric="comparison/step")

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _save_resolved_config(self, path):
        """Atomically save the fully resolved config alongside a model."""
        if not self.accelerator.is_main_process:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = self.config.unwrap() if isinstance(self.config, AccessTrackedConfig) else self.config
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        OmegaConf.save(config, temporary_path, resolve=True)
        os.replace(temporary_path, path)

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        self._sync_jepa_teachers_after_load = False
        self._resume_state_checkpoint = None

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                if os.path.isdir(resume_from_checkpoint):
                    self._resume_state_checkpoint = resume_from_checkpoint
                else:
                    self.model = self.load_pretrained_backbones(
                        self.model, self.resume_from_checkpoint, reload_modules=None
                    )
                    self._sync_jepa_teachers_after_load = bool(
                        getattr(self.model, "_jepa_teacher_keys_missing", False)
                    )
                    logger.warning(
                        "Legacy checkpoint contains model weights only; optimizer and RNG "
                        "state cannot be restored exactly."
                    )
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            self._sync_jepa_teachers_after_load = bool(
                getattr(self.model, "_jepa_teacher_keys_missing", False)
            )
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0 and self._resume_state_checkpoint is None:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save model, optimizer, scheduler, scaler, and RNG state on every rank."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        self.accelerator.save_state(checkpoint_path)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"Checkpoint saved at {checkpoint_path}")
            self._save_resolved_config(Path(checkpoint_path) / "config.yaml")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")
            if self.completed_steps > 0:
                smoke_checkpoint = Path(self.checkpoint_dir) / "steps_0"
                if smoke_checkpoint.is_dir():
                    shutil.rmtree(smoke_checkpoint)
                    logger.info("Removed step-0 smoke checkpoint after first periodic save")

        self.accelerator.wait_for_everyone()

    def _save_best_model(self, val_action_mse: float):
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            best_dir = Path(self.config.output_dir) / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model_path = best_dir / "pytorch_model.pt"
            temporary_model = best_dir / "pytorch_model.pt.tmp"
            torch.save(state_dict, temporary_model)
            os.replace(temporary_model, model_path)
            temporary_metrics = best_dir / "metrics.json.tmp"
            temporary_metrics.write_text(
                json.dumps(
                    {"step": self.completed_steps, "val_action_mse": val_action_mse},
                    indent=2,
                )
                + "\n"
            )
            os.replace(temporary_metrics, best_dir / "metrics.json")
            self._save_resolved_config(best_dir / "config.yaml")
        del state_dict
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and dist.get_rank() == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            metrics.update(comparison_metrics(metrics, self.completed_steps, last_lrs))
            wandb.log(metrics, step=self.completed_steps)
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vla_val_iter = iter(self.vla_val_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self.checkpoint_steps = checkpoint_milestones(
            self.config.trainer.max_train_steps,
            self.config.trainer.get("checkpoint_count", DEFAULT_CHECKPOINT_COUNT),
        )
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            completed_optimizer_step = self.accelerator.sync_gradients
            if completed_optimizer_step:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if completed_optimizer_step and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["timing/data"] = t_end_data - t_start_data
            step_metrics["timing/model"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps in self.checkpoint_steps:
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Evaluate deterministic held-out episodes and update the single best model."""
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()
        squared_error = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        count = torch.zeros((), dtype=torch.float64, device=self.accelerator.device)
        val_batches = int(self.config.trainer.get("val_batches", 8))
        device_index = self.accelerator.device.index
        fork_devices = [device_index] if device_index is not None else []
        with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(self.config.seed) + 10_000)
            for _ in range(val_batches):
                try:
                    examples = next(self.vla_val_iter)
                except StopIteration:
                    self.vla_val_iter = iter(self.vla_val_dataloader)
                    examples = next(self.vla_val_iter)
                output = model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=20)
                prediction = torch.as_tensor(output["normalized_actions"], device=self.accelerator.device)
                target = torch.as_tensor(np.asarray([item["action"] for item in examples]), device=prediction.device)
                mask = torch.as_tensor(
                    np.asarray([item.get("action_mask", np.ones_like(item["action"])) for item in examples]),
                    device=prediction.device,
                    dtype=torch.bool,
                )
                squared_error += ((prediction.double() - target.double()).square() * mask).sum()
                count += mask.sum()
        if dist.is_initialized():
            dist.all_reduce(squared_error)
            dist.all_reduce(count)
        mse = (squared_error / count.clamp_min(1)).item()
        step_metrics["val_action_mse"] = mse
        metrics_path = Path(self.config.output_dir) / "best_model" / "metrics.json"
        best_mse = float("inf")
        if metrics_path.is_file():
            best_mse = float(json.loads(metrics_path.read_text())["val_action_mse"])
        if mse < best_mse:
            self._save_best_model(mse)
        model.train(was_training)
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Checkpoint milestones = {self.checkpoint_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                loss_weights = (
                    unwrapped_model.jepa_loss_weights()
                    if hasattr(unwrapped_model, "jepa_loss_weights")
                    else {}
                )
                for loss_name, weight in loss_weights.items():
                    if loss_name in output_dict:
                        total_loss = total_loss + float(weight) * output_dict[loss_name]

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            optimizer_step_succeeded = not self.accelerator.optimizer_step_was_skipped
            if self.accelerator.sync_gradients and optimizer_step_succeeded:
                self.lr_scheduler.step()
                if hasattr(unwrapped_model, "update_jepa_teachers"):
                    unwrapped_model.update_jepa_teachers()
            self.optimizer.zero_grad()

        metrics = {
            "action_dit_loss": action_loss.item(),
            "total_loss": total_loss.item(),
        }
        for loss_name in ("tactile_loss", "state_jepa_loss", "vision_jepa_loss"):
            if loss_name in output_dict:
                metrics[loss_name] = output_dict[loss_name].item()
        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    accelerator = build_accelerator(cfg)
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader, vla_val_dataloader = prepare_data(
        cfg=cfg, accelerator=accelerator, output_dir=output_dir
    )
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        vla_val_dataloader=vla_val_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_收紧.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
