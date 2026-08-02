#!/usr/bin/env bash
# Creates (or reuses) the single project venv inside $MWW_DATA_DIR, so a pod
# that mounts the same persistent volume never re-installs torch and
# tensorflow from scratch -- the install only happens once per volume, and
# every pod after that just reattaches it and skips straight to training.
#
# Uses uv (not pip) -- already present on runpod/base images. Dramatically
# faster for repeated multi-GB torch/tensorflow reinstalls, and its `-e` from
# a local directory (not a Git URL -- uv doesn't support that, see the note
# in requirements.txt) is how the microwakeword packaging-bug fix works.
#
# (We previously tried baking requirements.txt into a custom Docker image
# instead. Reverted: the image pull cost on a fresh RunPod host wasn't
# actually smaller than a pip install, and unlike a persistent volume, a
# pulled image doesn't stick around when your pod lands on a different host
# next time. A venv on the volume is the one-time cost we actually wanted.)
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found on PATH. runpod/base images ship it at /bin/uv; install it yourself otherwise (https://astral.sh/uv)." >&2
  exit 1
fi

ROOTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${MWW_DATA_DIR:-./data}"
VENV_DIR="${DATA_DIR}/venv"
PIN_FILE="${VENV_DIR}/.requirements.sha256"

mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"  # resolve to an absolute path for the message below

if [[ -z "${MWW_DATA_DIR:-}" ]]; then
  echo "WARNING: \$MWW_DATA_DIR is not set -- defaulting to ${DATA_DIR}." >&2
  echo "         If this is a RunPod pod, that's almost certainly the wrong disk (probably the small root/container disk, not your Network Volume) -- set MWW_DATA_DIR before running this." >&2
fi
echo "Using MWW_DATA_DIR=${DATA_DIR}"

# uv's own download cache defaults to ~/.cache/uv on the ROOT filesystem,
# independent of where the venv itself lives -- redirect it onto the data
# dir too so a several-GB cache can't silently fill a small root disk even
# when MWW_DATA_DIR is set correctly. (Matches RunPod's own base image
# convention of pointing pip's equivalent PIP_CACHE_DIR at
# $RP_WORKSPACE/.cache/pip.)
export UV_CACHE_DIR="${DATA_DIR}/.cache/uv"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating venv at ${VENV_DIR}"
  # --python 3.10 is required, not cosmetic: uv's own Python-discovery
  # picked 3.12 by default on a RunPod pod (confirmed for real), unlike
  # `python3 -m venv`'s old incidental PATH-based resolution which always
  # landed on 3.10 there. audiomentations -> librosa -> numba -> llvmlite
  # 0.36.0 only builds on Python <3.10, so 3.12 fails outright at install
  # time. Pin explicitly instead of trusting whatever uv would pick.
  uv venv "${VENV_DIR}" --python 3.10
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

REQ_HASH="$(sha256sum "${ROOTDIR}/requirements.txt" | awk '{print $1}')"

if [[ ! -f "${PIN_FILE}" ]] || [[ "$(cat "${PIN_FILE}")" != "${REQ_HASH}" ]]; then
  echo "Installing/upgrading requirements"

  # Cleanup pass BEFORE the real install, not after: `uv pip install -r`
  # (like pip) never uninstalls a package that's been dropped from
  # requirements.txt, so anything we've deliberately excluded/replaced can
  # linger indefinitely across reinstalls. Two concrete cases hit for real:
  #   - piper-sample-generator (PyPI): replaced by a git clone (see the note
  #     in requirements.txt), but its stale audiomentations==0.33.0 pin
  #     causes a spurious resolver conflict warning on every reinstall.
  #   - webrtcvad (plain): both it and webrtcvad-wheels install a same-named
  #     `webrtcvad.py`, and neither pip nor uv has any idea the two collide --
  #     whichever installs/reinstalls LAST physically wins, silently. Plain
  #     webrtcvad actually won that race for real and broke the whole
  #     pipeline (its webrtcvad.py does `import pkg_resources`, which recent
  #     setuptools doesn't reliably bundle). Uninstalling both before
  #     reinstalling webrtcvad-wheels guarantees a clean, correct file
  #     regardless of whatever ordering corruption happened in a previous
  #     run -- `--upgrade` alone wouldn't fix this, since it's a no-op for a
  #     package that's already "satisfied" even if its files got clobbered
  #     by something else afterwards.
  #   - microwakeword (editable git install): pre-dates the pip->uv
  #     migration, when requirements.txt still had `-e git+URL#egg=...`. A
  #     venv created before that migration will have this leftover; sys.path
  #     ordering should make the new git-clone-on-PYTHONPATH approach win
  #     regardless, but no reason to leave stale, potentially-shadowing
  #     package metadata lying around when it's this cheap to remove.
  uv pip uninstall piper-sample-generator webrtcvad webrtcvad-wheels microwakeword >/dev/null 2>&1 || true

  # --upgrade matters here too, not just on first install: an unpinned
  # requirement line is "satisfied" by whatever's already installed, even a
  # stale version pulled in earlier by a *different* package's hard pin that
  # has since been removed from requirements.txt. Without --upgrade that
  # stale version silently survives every future reinstall. Hit this for
  # real with audiomentations (piper-sample-generator used to pin it to
  # 0.33.0; after removing that dependency, plain `install -r` left 0.33.0
  # in place since our own line was just "audiomentations").
  uv pip install --upgrade -r "${ROOTDIR}/requirements.txt"
  echo "${REQ_HASH}" > "${PIN_FILE}"
else
  echo "Reusing existing venv (requirements.txt unchanged since last install)"
fi

# Also install this repo's own package (mww_trainer) in editable mode so the
# scripts can `import mww_trainer.paths` etc. A local directory, not a Git
# URL, so uv's lack of editable-from-Git support (see requirements.txt)
# doesn't apply here.
uv pip show mww-trainer-scripts >/dev/null 2>&1 || uv pip install -e "${ROOTDIR}" --no-deps

echo "Venv ready: ${VENV_DIR}"
