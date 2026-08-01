#!/usr/bin/env bash
# Creates (or reuses) the single project venv inside $MWW_DATA_DIR, so a pod
# that mounts the same persistent volume never re-installs torch and
# tensorflow from scratch -- the install only happens once per volume, and
# every pod after that just reattaches it and skips straight to training.
#
# (We previously tried baking requirements.txt into a custom Docker image
# instead. Reverted: the image pull cost on a fresh RunPod host wasn't
# actually smaller than a pip install, and unlike a persistent volume, a
# pulled image doesn't stick around when your pod lands on a different host
# next time. A venv on the volume is the one-time cost we actually wanted.)
set -euo pipefail

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
