from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
CLIENT_SRC_DIR = ROOT_DIR / "packages" / "openpi-client" / "src"
for path in (SRC_DIR, CLIENT_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct: dict[str, Any]):
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(json_str, object_hook=object_hook)


class RoboTwinPolicyModel:
    def __init__(
        self,
        *,
        train_config_name: str,
        checkpoint_dir: str,
        default_prompt: str | None = None,
        pytorch_device: str | None = None,
    ):
        train_config = _config.get_config(train_config_name)
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )

    def get_action(self, obs: dict[str, Any]):
        return self.policy.infer(obs)

    def reset_model(self):
        return None


class PolicyServer:
    def __init__(self, model: RoboTwinPolicyModel, host: str, port: int):
        self.model = model
        self.host = host
        self.port = port

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(min(remaining, 4096))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _send_payload(sock: socket.socket, payload: dict[str, Any]):
        raw = numpy_to_json(payload).encode("utf-8")
        sock.sendall(len(raw).to_bytes(4, "big"))
        sock.sendall(raw)

    def _dispatch(self, request: dict[str, Any]):
        cmd = request.get("cmd")
        obs = request.get("obs")
        if cmd not in {"get_action", "reset_model"}:
            raise ValueError(f"Unsupported command: {cmd}")

        method = getattr(self.model, cmd)
        return method(obs) if obs is not None else method()

    def _handle_client(self, client_socket: socket.socket, addr: tuple[str, int]):
        with client_socket:
            while True:
                len_bytes = self._recv_exact(client_socket, 4)
                if len_bytes is None:
                    print(f"client disconnected: {addr}")
                    return

                request_size = int.from_bytes(len_bytes, "big")
                request_bytes = self._recv_exact(client_socket, request_size)
                if request_bytes is None:
                    print(f"client disconnected during request: {addr}")
                    return

                try:
                    request = json_to_numpy(request_bytes.decode("utf-8"))
                    result = self._dispatch(request)
                    self._send_payload(client_socket, {"ok": True, "result": result})
                except Exception as exc:
                    self._send_payload(
                        client_socket,
                        {
                            "ok": False,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )

    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"robotwin openpi server listening on {self.host}:{self.port}")

            while True:
                client_socket, addr = server_socket.accept()
                print(f"client connected: {addr}")
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr),
                    daemon=True,
                )
                thread.start()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="OpenPI train config name")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--pytorch-device", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    return parser.parse_args()


def main():
    args = parse_args()
    model = RoboTwinPolicyModel(
        train_config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    server = PolicyServer(model=model, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
