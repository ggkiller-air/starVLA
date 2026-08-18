# starVLA SONIC tactile training

This branch provides the three Table 1 modes from one codebase. Future observations are
training-only teacher targets; action prediction always conditions on the prompt, current
46-D state, current stereo pair, and current tactile packet only when tactile is enabled.

| Config | Current tactile | Future tactile | Future state | Future stereo |
|---|---:|---:|---:|---:|
| No Tactile | no | no | no | no |
| HTD | yes | yes | no | no |
| UniVLaT/JEPA | yes | yes | yes | yes |

The legacy `tactile_mode: input` remains available as a current-tactile-only ablation, but
it is not HTD. HTD uses `tactile_mode: dream` with both other dream flags disabled.

HTD is short for *Humanoid Transformer with Touch Dreaming* (arXiv:2604.13015). In this
port, HTD mode names the current-tactile fusion and future-tactile latent objective; it is
not a claim that starVLA reproduces the paper's complete policy and controller system.
Action and auxiliary-target masks exclude repeated episode-tail padding from every loss.

## Environment

```bash
cd /home/wzh/Projects/Uni_VLaT/starVLA
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

## Full training

The fixed configs contain 120k steps, four 30k-step checkpoints, four workers per rank,
and the measured throughput-optimal batch of 4 per GPU (global batch 16). HTD takes about
14-15 hours and processes 1.92M samples (about 21 dataset passes). This is a practical
fine-tuning budget rather than a forced sample-count match. Resume only if validation or
robot success is still improving. The configs use the faster `decord` backend; a 120-step
HTD run measured negligible prefetched data wait.

```bash
cd /home/wzh/Projects/Uni_VLaT/starVLA
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3

# No Tactile
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Sonic/train_files/starvla_train_sonic_notactile.yaml

# HTD: current tactile fusion plus future-tactile teacher
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Sonic/train_files/starvla_train_sonic_htd.yaml

# UniVLaT/JEPA: HTD plus future-state and future-stereo teachers
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Sonic/train_files/starvla_train_sonic_jepa.yaml
```

Checkpoints are written under
`results/Checkpoints/sonic_<mode>/checkpoints/steps_<step>_pytorch_model.pt`; the completed
model is in `results/Checkpoints/sonic_<mode>/final_model/`.

## Deploy through SONIC

The backend returns a finite `float32[40,78]` chunk laid out as 64 SONIC motion-token
values followed by 7 left-hand and 7 right-hand values. SONIC, not starVLA, decodes the
motion token into G1 whole-body control.

Terminal 1, start the starVLA websocket backend (port 8000):

```bash
cd /home/wzh/Projects/Uni_VLaT/starVLA
CUDA_VISIBLE_DEVICES=0 .venv/bin/python deployment/model_server/server_sonic_policy.py \
  --ckpt-path /home/shared/outputs/starVLA/sonic_htd_compute_matched_20260814_vla_remaining/best_model/pytorch_model.pt \
  --device cuda:0 --use-bf16 --port 8000
```

Do not pass the old `desk_sweep` unnormalization key. This checkpoint contains one key,
`unitree_g1_sonic`, which the server selects automatically when `--unnorm-key` is omitted.

Terminal 2, expose that backend through the Isaac-GR00T ZMQ PolicyServer (port 5550):

```bash
cd /home/wzh/Projects/Uni_VLaT/Isaac-GR00T
.venv/bin/python gr00t/eval/run_sonic_bridge_server.py \
  --backend-host 127.0.0.1 --backend-port 8000 \
  --host 0.0.0.0 --port 5550
```

Terminal 3, launch the shared controller and inference client:

```bash
cd /home/wzh/Projects/Uni_VLaT/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_inference.py \
  --policy-host 127.0.0.1 --policy-port 5550 \
  --policy-timeout-ms 60000 \
  --camera-host 192.168.123.164 \
  --tactile-zmq-host 192.168.123.164 \
  --prompt "carry the bucket"
```

For a No Tactile checkpoint, omit `--tactile-zmq-host` and add `--no-use-tactile`.
The verified HTD best checkpoint runs in bf16 on one 24 GB RTX 4090.
