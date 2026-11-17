ARG POSE6D_VISION_BASE_IMAGE=nvcr.io/nvidian/bundlesdf@sha256:534a24c4fb89c89ba3aefbbaef97dce9b693656e27f46bf55ebb85e45c2842dd
FROM ${POSE6D_VISION_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/pose6d_vision \
    YOLO_CONFIG=/etc/pose6d/pose6d_yolo.yaml \
    YOLO_WEIGHTS=/models/yolov8n.pt \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    XDG_CACHE_HOME=/tmp/cache
WORKDIR /opt/pose6d_vision
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY yolo/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY common /opt/pose6d_vision/common
COPY yolo/server.py /opt/pose6d_vision/yolo/server.py
RUN mkdir -p /opt/pose6d_vision/yolo /models /tmp/ultralytics
EXPOSE 8093
ENTRYPOINT ["python3", "-m", "yolo.server"]
CMD ["--host", "0.0.0.0", "--port", "8093"]
