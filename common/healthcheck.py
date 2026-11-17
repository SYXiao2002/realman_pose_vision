"""Docker health check using the same WebSocket hello as the bridge."""

import argparse

from websockets.sync.client import connect

from .wire import unpackb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--service", required=True)
    args = parser.parse_args()
    with connect(f"ws://127.0.0.1:{args.port}", open_timeout=2, close_timeout=1) as websocket:
        hello = unpackb(websocket.recv(timeout=2))
    if hello.get("type") != "hello" or hello.get("ready") is not True or hello.get("service") != args.service:
        raise SystemExit("service is not ready")


if __name__ == "__main__":
    main()
