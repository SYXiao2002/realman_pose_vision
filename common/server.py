"""Single-request, bounded synchronous WebSocket inference service."""

import logging
import threading
import time

from websockets.sync.server import serve
from websockets.exceptions import ConnectionClosed

from .wire import MAX_MESSAGE_BYTES, packb, unpackb

LOG = logging.getLogger("pose6d_vision")


class InferenceServer:
    def __init__(self, *, host, port, service, hello, handler, timeout_sec=90.0):
        self.host = host
        self.port = port
        self.service = service
        self.hello = {"type": "hello", "version": 1, "ready": True, "service": service, **hello}
        self.handler = handler
        self.timeout_sec = timeout_sec
        self.busy = threading.Lock()

    def handle(self, websocket):
        websocket.send(packb(self.hello))
        request = None
        try:
            try:
                raw = websocket.recv(timeout=self.timeout_sec)
            except ConnectionClosed:
                return
            started = time.perf_counter()
            request = unpackb(raw)
            if not self.busy.acquire(blocking=False):
                raise RuntimeError("inference service is busy")
            try:
                response = self.handler(request)
            finally:
                self.busy.release()
            response["service_ms"] = (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            LOG.exception("%s request failed", self.service)
            response = {
                "type": "error",
                "version": 1,
                "request_id": request.get("request_id") if isinstance(request, dict) else None,
                "observation_id": request.get("observation_id") if isinstance(request, dict) else None,
                "code": "busy" if isinstance(exc, RuntimeError) and "busy" in str(exc) else "inference_failed",
                "message": str(exc)[:512],
            }
        try:
            websocket.send(packb(response))
        except Exception:
            LOG.exception("%s response send failed", self.service)

    def run(self):
        LOG.info("%s ready on %s:%s", self.service, self.host, self.port)
        with serve(self.handle, self.host, self.port, max_size=MAX_MESSAGE_BYTES, compression=None) as server:
            server.serve_forever()
