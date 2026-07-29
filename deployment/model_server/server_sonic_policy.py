"""Serve a starVLA JEPA checkpoint through the canonical SONIC contract."""

import argparse
import logging

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.sonic_policy import SonicPolicyAdapter
from deployment.model_server.tools.sonic_websocket_policy_server import (
    SonicWebsocketPolicyServer,
)


def main(args) -> None:
    policy = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device=args.device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
    )
    adapter = SonicPolicyAdapter(policy, unnorm_key=args.unnorm_key)
    logging.info("SONIC metadata: %s", adapter.metadata)
    SonicWebsocketPolicyServer(adapter, host=args.host, port=args.port).serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser(description="starVLA SONIC websocket server")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--unnorm-key", default=None)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
