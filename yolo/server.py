"""YOLOv8 detector service used by the single-frame pose6d workflow."""

import argparse
import hashlib
import logging
import math
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from common.server import InferenceServer, LOG


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("YOLO config must be a mapping")
    weight = Path(config["weights_path"])
    if not weight.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weight}")
    actual_hash = hashlib.sha256(weight.read_bytes()).hexdigest()
    if actual_hash != config["weights_sha256"]:
        raise ValueError("YOLO weights SHA-256 does not match config")
    threshold = config["confidence_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be from 0 through 1")
    max_detections = config["max_detections"]
    if type(max_detections) is not int or not 1 <= max_detections <= 32:
        raise ValueError("max_detections must be from 1 through 32")
    names = config["class_names"]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("class_names must be a non-empty string list")
    return config, weight, actual_hash


def validate_request(request, maximum_detections, maximum_pixels):
    if not isinstance(request, dict) or request.get("type") != "detect" or request.get("version") != 1:
        raise ValueError("expected version 1 detect request")
    if set(request) != {"type", "version", "request_id", "observation_id", "rgb",
                        "confidence_threshold", "max_detections"}:
        raise ValueError("detect request contains unexpected fields")
    for key in ("request_id", "observation_id"):
        value = request.get(key)
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ValueError(f"invalid {key}")
    rgb = request.get("rgb")
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    if not 0 < rgb.shape[0] <= 4096 or not 0 < rgb.shape[1] <= 4096:
        raise ValueError("invalid image dimensions")
    if rgb.shape[0] * rgb.shape[1] > maximum_pixels:
        raise ValueError("image exceeds the configured pixel limit")
    threshold = request.get("confidence_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (float, int))
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("invalid confidence_threshold")
    max_detections = request.get("max_detections")
    if type(max_detections) is not int or not 1 <= max_detections <= maximum_detections:
        raise ValueError("invalid max_detections")
    return rgb, float(threshold), max_detections


class Detector:
    def __init__(self, config, weight, weights_hash):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the YOLO service")
        from ultralytics import YOLO

        self.config = config
        self.weights_hash = weights_hash
        self.model = YOLO(str(weight))
        names = self.model.names
        names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)
        if names != config["class_names"]:
            raise ValueError(f"YOLO class names differ from configured model identity: {names}")
        self.class_names = names
        self.device = "0"
        self.model.predict(
            source=np.zeros((320, 320, 3), dtype=np.uint8),
            conf=config["confidence_threshold"],
            max_det=config["max_detections"],
            device=self.device,
            verbose=False,
        )
        LOG.info("YOLO weights ready: sha256=%s names=%s", weights_hash, names)

    def infer(self, request):
        rgb, threshold, max_detections = validate_request(
            request, self.config["max_detections"], self.config["max_image_pixels"])
        image = np.ascontiguousarray(rgb[:, :, ::-1])
        started = time.perf_counter()
        with torch.inference_mode():
            result = self.model.predict(
                source=image,
                conf=threshold,
                max_det=max_detections,
                device=self.device,
                verbose=False,
            )[0]
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0
        detections = []
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
            confidences = boxes.conf.detach().cpu().numpy()
            height, width = rgb.shape[:2]
            for index, (box, class_id, confidence) in enumerate(zip(xyxy, classes, confidences)):
                x1, y1, x2, y2 = box.tolist()
                bbox = [
                    max(0, min(width, int(math.floor(x1)))),
                    max(0, min(height, int(math.floor(y1)))),
                    max(0, min(width, int(math.ceil(x2)))),
                    max(0, min(height, int(math.ceil(y2)))),
                ]
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                detections.append({
                    "detection_id": str(index),
                    "bbox": bbox,
                    "class_id": int(class_id),
                    "class_name": self.class_names[int(class_id)],
                    "confidence": float(confidence),
                })
        return {
            "type": "result",
            "version": 1,
            "request_id": request["request_id"],
            "observation_id": request["observation_id"],
            "model_sha256": self.weights_hash,
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "detections": detections,
            "yolo_inference_ms": inference_ms,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--config", default="/etc/pose6d/pose6d_yolo.yaml")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config, weight, weights_hash = load_config(args.config)
    detector = Detector(config, weight, weights_hash)
    InferenceServer(
        host=args.host,
        port=args.port,
        service="pose6d_yolo",
        hello={"model_sha256": weights_hash, "class_names": detector.class_names, "device": "cuda:0"},
        handler=detector.infer,
        timeout_sec=args.timeout,
    ).run()


if __name__ == "__main__":
    main()
