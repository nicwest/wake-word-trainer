# Bakes the full pip environment (torch + tensorflow + everything else in
# requirements.txt) into the image at build time, so a fresh RunPod pod skips
# straight to "pull image, mount data volume, train" -- no pip install wait
# on the very first boot in a datacenter.
#
# Base: runpod/base ships Python 3.9-3.13 (3.12 symlinked as default) and
# build-essential/cmake/ffmpeg/uv, on top of plain Ubuntu 22.04 -- NOT a
# cuda-tagged variant. We don't need the CUDA toolkit for anything ourselves
# (torch/tensorflow bring their own pip-installed CUDA runtime libs, see
# requirements.txt), and RunPod's own build config
# (github.com/runpod/containers official-templates/base/docker-bake.hcl)
# shows the cuda-tagged variants are built FROM nvidia/cuda:*-cudnn-devel-*,
# which bakes in a hard `NVIDIA_REQUIRE_CUDA=cuda>=X.Y` driver-version gate
# checked by the nvidia-container-cli prestart hook at container start.
# Learned this the hard way: a cuda1281-tagged base refused to start on a
# pod whose host driver didn't satisfy cuda>=12.8, even though nothing in
# this image actually uses the system CUDA toolkit. Plain ubuntu2204 has no
# such gate -- GPU access still works via the standard device/driver-library
# passthrough RunPod sets up at the container-runtime level, independent of
# whether the image is nvidia/cuda-derived.
FROM runpod/base:1.1.0-ubuntu2204

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
