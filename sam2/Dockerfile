ARG POSE6D_VISION_BASE_IMAGE=nvcr.io/nvidian/bundlesdf@sha256:534a24c4fb89c89ba3aefbbaef97dce9b693656e27f46bf55ebb85e45c2842dd
FROM ${POSE6D_VISION_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/pose6d_vision:/opt/sam2 \
    XDG_CACHE_HOME=/tmp/cache \
    TORCH_CUDA_ARCH_LIST=8.9 FORCE_CUDA=1 MAX_JOBS=2
WORKDIR /opt/pose6d_vision
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY sam2/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
COPY common /opt/pose6d_vision/common
COPY sam2/server.py /opt/pose6d_vision/sam2_service/server.py
RUN mkdir -p /opt/pose6d_vision/sam2_service /opt/sam2
COPY --from=sam2_source /sam2 /opt/sam2/sam2
COPY --from=sam2_source /setup.py /pyproject.toml /README.md /opt/sam2/
WORKDIR /opt/sam2
RUN SAM2_BUILD_ALLOW_ERRORS=0 python3 -m pip install --no-cache-dir --no-build-isolation -e .
WORKDIR /opt/pose6d_vision
EXPOSE 8094
ENTRYPOINT ["python3", "-m", "sam2_service.server"]
CMD ["--host", "0.0.0.0", "--port", "8094"]
