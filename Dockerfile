FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /requirements.txt

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python-is-python3 \
    python3 \
    python3-pip \
  && python -m pip install --no-cache-dir -r requirements.txt \
  && rm -f requirements.txt \
  && rm -rf /var/lib/apt/lists/*
