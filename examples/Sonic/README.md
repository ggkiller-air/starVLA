# SONIC Deployment

The JEPA checkpoint is served through the same flat websocket contract used by
the Isaac-GR00T SONIC bridge:

```bash
python -m deployment.model_server.server_sonic_policy \
  --ckpt-path /path/to/checkpoints/model.pt \
  --device cuda \
  --use-bf16 \
  --port 8000
```

The server accepts current stereo RGB, the canonical 46-dimensional G1 state,
the prompt, and one concatenated `uint8[768]` tactile frame. It normalizes state and
unnormalizes the `40 x 78` output with the exact training transforms before
returning the SONIC motion-token/hand action chunk. The websocket handshake
uses protocol `sonic_vla_v1` and is compatible with Isaac-GR00T's bridge
PolicyServer.
