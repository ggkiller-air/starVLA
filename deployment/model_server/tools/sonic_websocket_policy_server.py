"""Openpi-compatible websocket server for the canonical SONIC adapter."""

import asyncio
import logging
import traceback

import websockets.asyncio.server
import websockets.frames

from . import msgpack_numpy


class SonicWebsocketPolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._policy = policy
        self._host = host
        self._port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logging.info("SONIC connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._policy.metadata))
        while True:
            try:
                observation = msgpack_numpy.unpackb(await websocket.recv())
                result = self._policy.infer(observation)
                await websocket.send(packer.pack(result))
            except websockets.ConnectionClosed:
                logging.info("SONIC connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="SONIC policy inference failed",
                )
                raise
