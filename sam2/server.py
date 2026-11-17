"""SAM2.1 Hiera Tiny box-prompt segmentation service."""

import argparse
import hashlib
import logging
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from common.server import InferenceServer, LOG


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SAM2 config must be a mapping")
    checkpoint = Path(config["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_hash != config["checkpoint_sha256"]:
        raise ValueError("SAM2 checkpoint SHA-256 does not match config")
    if config.get("model_config") != "configs/sam2.1/sam2.1_hiera_t.yaml":
        raise ValueError("only SAM2.1 Hiera Tiny is supported by this service")
    maximum = config.get("max_detections")
    if type(maximum) is not int or not 1 <= maximum <= 32:
        raise ValueError("max_detections must be from 1 through 32")
    return config, checkpoint, actual_hash


def validate_request(request, maximum_detections, maximum_pixels):
    if not isinstance(request, dict) or request.get("type") != "segment" or request.get("version") != 1:
        raise ValueError("expected version 1 segment request")
    if set(request) != {"type", "version", "request_id", "observation_id", "rgb", "detections"}:
        raise ValueError("segment request contains unexpected fields")
    for key in ("request_id", "observation_id"):
        value = request.get(key)
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ValueError(f"invalid {key}")
    rgb = request.get("rgb")
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    height, width = rgb.shape[:2]
    if not 0 < height <= 4096 or not 0 < width <= 4096:
        raise ValueError("invalid image dimensions")
    if height * width > maximum_pixels:
        raise ValueError("image exceeds the configured pixel limit")
    detections = request.get("detections")
    if not isinstance(detections, list) or len(detections) > maximum_detections:
        raise ValueError("invalid detection count")
    seen = set()
    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError("invalid detection item")
        if set(detection) != {"detection_id", "bbox", "class_id", "class_name", "confidence"}:
            raise ValueError("detection contains unexpected fields")
        detection_id = detection.get("detection_id")
        bbox = detection.get("bbox")
        if not isinstance(detection_id, str) or detection_id in seen:
            raise ValueError("detection IDs must be unique strings")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(type(value) is not int for value in bbox):
            raise ValueError("bbox must contain four integers")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bbox is outside the image")
        seen.add(detection_id)
    return rgb, detections


class Segmenter:
    def __init__(self, config, checkpoint, checkpoint_hash):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the SAM2 service")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.config = config
        self.checkpoint_hash = checkpoint_hash
        model = build_sam2(config["model_config"], str(checkpoint), device="cuda")
        self.predictor = SAM2ImagePredictor(model)
        LOG.info("SAM2.1 Tiny checkpoint ready: sha256=%s", checkpoint_hash)

    def infer(self, request):
        rgb, detections = validate_request(
            request, self.config["max_detections"], self.config["max_image_pixels"])
        started = time.perf_counter()
        results = []
        image_ms = 0.0
        predict_total_ms = 0.0
        if detections:
            image_started = time.perf_counter()
            with torch.inference_mode():
                self.predictor.set_image(np.ascontiguousarray(rgb))
            torch.cuda.synchronize()
            image_ms = (time.perf_counter() - image_started) * 1000.0
            for detection in detections:
                predict_started = time.perf_counter()
                try:
                    with torch.inference_mode():
                        masks, scores, _ = self.predictor.predict(
                            box=np.asarray(detection["bbox"], dtype=np.float32),
                            multimask_output=False,
                        )
                    torch.cuda.synchronize()
                    mask = np.asarray(masks[0])
                    if mask.shape != rgb.shape[:2]:
                        raise ValueError("SAM2 mask dimensions differ from input RGB")
                    mask = np.ascontiguousarray(mask != 0, dtype=np.uint8)
                    pixels = int(np.count_nonzero(mask))
                    if pixels == 0:
                        raise ValueError("SAM2 returned an empty mask")
                    score = float(np.asarray(scores).reshape(-1)[0])
                    elapsed_ms = (time.perf_counter() - predict_started) * 1000.0
                    predict_total_ms += elapsed_ms
                    results.append({
                        "detection_id": detection["detection_id"],
                        "status": "ok",
                        "sam_score": score,
                        "mask_pixels": pixels,
                        "sam_predict_ms": elapsed_ms,
                        "mask": mask,
                    })
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - predict_started) * 1000.0
                    predict_total_ms += elapsed_ms
                    results.append({
                        "detection_id": detection["detection_id"],
                        "status": "error",
                        "message": str(exc)[:256],
                        "sam_predict_ms": elapsed_ms,
                    })
        return {
            "type": "result",
            "version": 1,
            "request_id": request["request_id"],
            "observation_id": request["observation_id"],
            "model_sha256": self.checkpoint_hash,
            "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "results": results,
            "sam_image_ms": image_ms,
            "sam_predict_ms_total": predict_total_ms,
            "sam_inference_ms_total": image_ms + predict_total_ms,
            "sam_request_ms": (time.perf_counter() - started) * 1000.0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--config", default="/etc/pose6d/pose6d_sam2.yaml")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config, checkpoint, checkpoint_hash = load_config(args.config)
    segmenter = Segmenter(config, checkpoint, checkpoint_hash)
    InferenceServer(
        host=args.host,
        port=args.port,
        service="pose6d_sam2",
        hello={"model_sha256": checkpoint_hash, "model_config": config["model_config"], "device": "cuda:0"},
        handler=segmenter.infer,
        timeout_sec=args.timeout,
    ).run()


if __name__ == "__main__":
    main()
