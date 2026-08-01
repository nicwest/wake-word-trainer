# Bakes the full pip environment (torch + tensorflow + everything else in
# requirements.txt) into the image at build time, so a fresh RunPod pod skips
# straight to "pull image, mount data volume, train" -- no pip install wait
# on the very first boot in a datacenter.
#
# Base: runpod/base ships Python 3.9-3.13 (3.12 symlinked as default),
# build-essential/cmake/ffmpeg, and uv, on top of CUDA 12.8.1/Ubuntu 22.04.
# We don't need the CUDA toolkit for anything ourselves (torch/tensorflow
# bring their own pip-installed CUDA runtime libs, see requirements.txt) but
# RunPod's GPU pods expect a cuda-tagged base for driver passthrough to work
# cleanly.
FROM runpod/base:1.1.0-cuda1281-ubuntu2204

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
 && python -m pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY mww_trainer ./mww_trainer
RUN python -m pip install --no-cache-dir -e . --no-deps

COPY scripts ./scripts
COPY train.sh .

# Tells train.sh / 00_setup_venv.sh that dependencies are already installed
# at the system level -- skip creating/populating a venv on $MWW_DATA_DIR.
ENV MWW_TRAINER_PREBUILT=1
ENV MWW_DATA_DIR=/workspace/data

CMD ["/bin/bash"]
