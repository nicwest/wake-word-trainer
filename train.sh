#!/usr/bin/env bash
# End-to-end: setup venv -> fetch shared assets -> generate samples ->
# build features -> train -> export. Every step is idempotent, so re-running
# after a failure (or on a fresh pod with the same $MWW_DATA_DIR mounted)
# just picks up where it left off.
#
# Usage:
#   MWW_DATA_DIR=/workspace/data ./train.sh "hey wild rider" [--skip-fetch] [--skip-generate] [--skip-features] [--force-uploaded]
#
# After uploading new real_samples/false_positive_samples with
# upload_samples.sh, the fast path to retrain on them is:
#   ./train.sh "hey wild rider" --skip-fetch --skip-generate --force-uploaded
# --force-uploaded rebuilds only the small real_samples/false_positive_samples
# features (03_build_features.py normally skips already-built features
# entirely, so newly uploaded files are otherwise silently ignored) without
# also redoing the much larger generated_samples set.
set -euo pipefail

ROOTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"<wake word phrase>\" [--skip-fetch] [--skip-generate] [--skip-features] [--force-uploaded]" >&2
  exit 1
fi

WAKE_WORD="$1"
shift

SKIP_FETCH=false
SKIP_GENERATE=false
SKIP_FEATURES=false
FORCE_UPLOADED=false
for arg in "$@"; do
  case "$arg" in
    --skip-fetch) SKIP_FETCH=true ;;
    --skip-generate) SKIP_GENERATE=true ;;
    --skip-features) SKIP_FEATURES=true ;;
    --force-uploaded) FORCE_UPLOADED=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

export MWW_DATA_DIR="${MWW_DATA_DIR:-./data}"
: "${MAX_SAMPLES:=3000}"
: "${TRAINING_STEPS:=10000}"

echo "== data dir: ${MWW_DATA_DIR} =="

source "${ROOTDIR}/scripts/00_setup_venv.sh"

if [[ "${SKIP_FETCH}" == false ]]; then
  python3 "${ROOTDIR}/scripts/01_fetch_assets.py"
fi

if [[ "${SKIP_GENERATE}" == false ]]; then
  python3 "${ROOTDIR}/scripts/02_generate_samples.py" "${WAKE_WORD}" --max-samples "${MAX_SAMPLES}"
fi

if [[ "${SKIP_FEATURES}" == false ]]; then
  if [[ "${FORCE_UPLOADED}" == true ]]; then
    python3 "${ROOTDIR}/scripts/03_build_features.py" "${WAKE_WORD}" --force-uploaded
  else
    python3 "${ROOTDIR}/scripts/03_build_features.py" "${WAKE_WORD}"
  fi
fi

python3 "${ROOTDIR}/scripts/04_train.py" "${WAKE_WORD}" --training-steps "${TRAINING_STEPS}"
python3 "${ROOTDIR}/scripts/05_export.py" "${WAKE_WORD}"

echo "== done: $(python3 -c "import sys; sys.path.insert(0, '${ROOTDIR}'); from mww_trainer import paths; print(paths.output_dir('${WAKE_WORD}'))") =="
