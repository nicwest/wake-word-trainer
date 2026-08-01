#!/usr/bin/env bash
# Creates (or reuses) the single project venv inside $MWW_DATA_DIR, so a pod
# that mounts the same persistent volume/bucket sync never re-installs torch
# and tensorflow from scratch.
#
# Skipped entirely when MWW_TRAINER_PREBUILT=1 (set by the Dockerfile): in
# that case requirements.txt was already installed into the image's system
# Python at build time, so there's nothing to do here -- `python3`/`pip`
# already resolve to an environment with everything installed.
set -euo pipefail

if [[ "${MWW_TRAINER_PREBUILT:-0}" == "1" ]]; then
  echo "MWW_TRAINER_PREBUILT=1: dependencies are baked into the image, skipping venv setup."
  return 0 2>/dev/null || exit 0
fi

ROOTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${MWW_DATA_DIR:-./data}"
VENV_DIR="${DATA_DIR}/venv"
PIN_FILE="${VENV_DIR}/.requirements.sha256"

mkdir -p "${DATA_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating venv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

REQ_HASH="$(sha256sum "${ROOTDIR}/requirements.txt" | awk '{print $1}')"

if [[ ! -f "${PIN_FILE}" ]] || [[ "$(cat "${PIN_FILE}")" != "${REQ_HASH}" ]]; then
  echo "Installing/upgrading requirements"
  pip install -U pip setuptools wheel
  pip install -r "${ROOTDIR}/requirements.txt"
  echo "${REQ_HASH}" > "${PIN_FILE}"
else
  echo "Reusing existing venv (requirements.txt unchanged since last install)"
fi

# Also install this repo's own package (mww_trainer) in editable mode so the
# scripts can `import mww_trainer.paths` etc.
pip show mww-trainer-scripts >/dev/null 2>&1 || pip install -e "${ROOTDIR}" --no-deps

echo "Venv ready: ${VENV_DIR}"
