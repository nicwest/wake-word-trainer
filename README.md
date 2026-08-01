# mww-trainer

A minimal microWakeWord training harness, built by stripping the [reference
notebook](https://github.com/kahrendt/microWakeWord/blob/main/notebooks/basic_training_notebook.ipynb)
and [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
down to scripts, and cutting everything that isn't needed just to produce a
`.tflite` + manifest JSON for ESPHome's `micro_wake_word` component.

Compared to [TaterTotterson/microWakeWord-Trainer-Nvidia-Docker](https://github.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker)
(which this replaces for our own use), this intentionally does **not** include:

- the FastAPI/uvicorn trainer UI, satellite auto-training loop, or STT
  (faster-whisper / silero-vad / onnx-asr) -- none of that trains a model
- a second, separate venv for that UI stack
- Piper voices for every language -- only the one TTS generator checkpoint we
  actually use
- the full FSD50K/AudioSet/FMA/WHAM/CHiME battery -- see "Data sources" below
  for what's used instead and why

Everything that should survive between sessions -- downloaded assets and
trained models -- lives under one directory, `$MWW_DATA_DIR`. Point that at a
RunPod persistent Network Volume (or a synced bucket) and a fresh pod skips
straight to training with no re-bootstrap.

The Python environment itself (torch + tensorflow + everything else in
`requirements.txt`) is baked into a Docker image at build time instead of
living on that persisted volume -- see "Docker image" below. That's the
difference between "pull an image" and "pip install several GB" on a pod
that's never seen this volume before.

## Docker image

```sh
docker pull ghcr.io/nicwest/wake-word-trainer:latest
```

(The image name is `ghcr.io/<github-org-or-user>/<repo-name>` via the publish
workflow in `.github/workflows/docker-publish.yml`, which builds on every
push to `main` and on `v*` tags -- so it'll auto-follow if this repo is ever
renamed or forked elsewhere.)

Run it on a RunPod pod (or locally) with the data dir mounted:

```sh
docker run --gpus all -it \
  -v /workspace/data:/workspace/data \
  ghcr.io/nicwest/wake-word-trainer:latest \
  ./train.sh "hey wild rider"
```

Inside the image, `MWW_TRAINER_PREBUILT=1` is already set, so `train.sh`
skips venv creation/pip install entirely and uses the image's system Python
directly -- only `01_fetch_assets.py`'s downloads and `work/<wake_word>/`
touch the mounted volume.

To build and push it yourself instead of waiting on CI:

```sh
docker build -t ghcr.io/nicwest/wake-word-trainer:latest .
docker push ghcr.io/nicwest/wake-word-trainer:latest
```

First push: GHCR packages default to private and linked-to-the-repo -- go to
https://github.com/nicwest/wake-word-trainer/pkgs/container/wake-word-trainer/settings
and set visibility to public if you want RunPod pods to pull it without a
registry login.

## Layout

```
$MWW_DATA_DIR/
  venv/                        only created outside the Docker image (bare-metal/local use); reused if requirements.txt is unchanged
  assets/                      shared across every wake word you train
    piper/                     LibriTTS-R generator checkpoint (~200MB)
    rir/                       MIT environmental impulse responses (~8MB)
    background/                curated background-noise subset (audioset clips + fma_xs)
    negative_features/         pre-extracted negative spectrogram sets (speech, dinner_party, dinner_party_eval, no_speech)
  work/<wake_word_slug>/       one subtree per phrase you train
    generated_samples/         raw TTS positives
    real_samples/               real (non-synthetic) positives you drop in yourself -- see below
    features/{training,validation,testing}/         features for generated_samples/
    real_features/{training,validation,testing}/     features for real_samples/, if any
    training_parameters.yaml
    trained_models/
    output/<slug>.tflite, <slug>.json
```

### Real positive samples (e.g. captured false negatives)

TTS-generated samples are the bulk of the positive training data, but real
device-captured recordings -- especially false negatives (times the wake word
was actually spoken but missed) -- are the highest-value signal for fixing
recall on your actual hardware/voice/room, since they reflect the real
deployment conditions instead of Piper's synthetic voices.

Drop them in `work/<wake_word_slug>/real_samples/` (wav/flac/mp3/ogg, any
sample rate -- `Clips` resamples to 16kHz automatically) and re-run
`03_build_features.py`. They get their own augmentation + feature pipeline
(`real_features/`) and their own entry in `training_parameters.yaml` with a
higher default sampling weight (4.0 vs. generated's 2.0, via
`04_train.py --real-sampling-weight`) rather than being merged into the TTS
set and diluted -- there are usually far fewer of them, so upweighting keeps
them from getting drowned out in each training batch. If there's nothing in
`real_samples/` yet, both scripts skip that half silently and behave exactly
as before.

With only a handful of clips, the automatic train/validation/testing split
can fail outright (not enough clips to carve out a slice) -- that's expected
until you've collected enough to be worth splitting; the error message says
so explicitly rather than crashing cryptically.

## Quick start

Via Docker (recommended for RunPod -- see above for the full `docker run`):

```sh
export MWW_DATA_DIR=/workspace/data   # wherever your persistent volume/bucket mount is
./train.sh "hey wild rider"
```

Bare-metal/local, without the image (creates a venv under `$MWW_DATA_DIR` on
first run, same idempotent skip-if-unchanged behavior):

```sh
export MWW_DATA_DIR=./data
./train.sh "hey wild rider"
```

Individual steps (useful for iterating on one wake word without re-touching
shared assets):

```sh
source scripts/00_setup_venv.sh   # no-op inside the Docker image
python3 scripts/01_fetch_assets.py               # once, shared by all wake words
python3 scripts/02_generate_samples.py "hey wild rider" --max-samples 2000
python3 scripts/03_build_features.py "hey wild rider"
python3 scripts/04_train.py "hey wild rider" --training-steps 10000
python3 scripts/05_export.py "hey wild rider" --author "you" --website "https://example.com"
```

## Data sources (and what's deliberately smaller)

| Purpose | Reference recipe | This harness |
|---|---|---|
| Positive samples | Piper LibriTTS-R generator | same |
| Room reverb | BIRD RIR corpus (large) | MIT environmental impulse responses (270 short clips) |
| Background noise | full FSD50K + AudioSet + FMA + WHAM + CHiME | one AudioSet balanced-train shard + FMA "extra small" |
| Negative/false-accept training data | pre-extracted `speech`/`dinner_party`/`dinner_party_eval`/`no_speech` spectrograms | same -- no smaller substitute without giving up false-accept robustness |

The background-noise trim is the main lever if disk size still matters more
than model robustness: `01_fetch_assets.py --skip-background` skips it
entirely (fast smoke test, worse false-accept rate), or pass
`--audioset-clips 5000` (etc.) to stream down more clips for more diversity.

## Disk footprint (measured, not guessed)

Sizes below came from HEAD requests / HF API / PyPI metadata against the
actual URLs this harness downloads (checked 2026-08-01), not from memory.

**Docker image**: ~6-7GB pulled once (base image + baked-in requirements.txt),
cached by the registry/datacenter -- doesn't count against `$MWW_DATA_DIR`
at all since it never touches the volume.

**`$MWW_DATA_DIR` (Docker path -- no `venv/`)**:

| Component | On-disk size |
|---|---|
| `assets/piper/` (LibriTTS-R generator checkpoint) | ~0.2 GB |
| `assets/rir/` (MIT impulse responses, already 16kHz) | ~0.01 GB |
| `assets/background/audioset/` (2000 clips, default) | ~0.6 GB |
| `assets/background/fma/` (fma_xs, converted to 16kHz PCM) | ~0.3-0.4 GB |
| `assets/negative_features/` (4 extracted zips, ~5.7GB compressed) | ~6-7 GB |
| `work/<wake_word>/` (samples + features + checkpoints + output) | ~0.3-0.8 GB per phrase |

**Total for shared assets + one wake word: roughly 7-8 GB.** Bare-metal/local
use (no Docker image) adds a `venv/` of torch+tensorflow+shared cu12 CUDA
libs, ~5-6GB, back on top of that (~13-15GB total) -- baking it into the
image instead is most of the point of building the image.

Give the volume real headroom beyond either total: extraction briefly needs
the zip *and* its extracted contents on disk at once (the negative-features
step peaks around 12GB transiently before cleanup), and TF checkpoints
accumulate during a run. **Size the RunPod volume at 20-25GB with the Docker
image (30-40GB without it)**, not the bare total above.

The `nvidia-*-cu12` pin in `requirements.txt` matters for both the image and
venv numbers: letting torch resolve to its latest version pulls
`nvidia-*-cu13` packages instead, which don't overlap with tensorflow's
`cu12` requirement -- that would install two full CUDA library sets side by
side (~2-3GB extra) instead of one shared set.

## RunPod persistence

Everything reusable is `$MWW_DATA_DIR`. Two ways to keep it warm between
sessions -- pick based on whether you want to stay in one datacenter or shop
for GPU availability each time:

- **Network Volume**: create one, mount it at `/workspace/data` on every pod,
  set `MWW_DATA_DIR=/workspace/data`. No sync step, ever. Locks you to that
  volume's datacenter and one pod at a time.
- **Bucket sync**: `rclone sync` `$MWW_DATA_DIR` to/from R2/B2/S3 at the start
  and end of each pod session -- pod-to-bucket only, so it runs at datacenter
  speed regardless of your home connection.

## Known gaps / next steps

- `05_export.py --tensor-arena-size` is a starting guess (32000), not
  computed from the model. ESPHome will fail to load the model if it's too
  small -- bump it and re-flash if that happens.
- No adversarial/similar-phrase negative generation (openWakeWord does this
  in the full microWakeWord recipe). Worth adding if false-accepts on
  phonetically similar words are a problem.
- Not tested end-to-end yet -- this was built from reading the reference
  notebook/source, not by running a full GPU training pass. Treat the first
  RunPod run as the real integration test.
- First real RunPod attempt hit `nvidia-container-cli: requirement error:
  unsatisfied condition: cuda>=12.8` -- the original `cuda1281`-tagged base
  image baked in a driver-version gate this pod's host didn't satisfy, even
  though nothing in the image uses system CUDA. Fixed by switching to the
  plain (non-cuda-tagged) `runpod/base:1.1.0-ubuntu2204`; see the comment at
  the top of `Dockerfile`. Not yet re-verified on a pod.
- Training itself (the actual GPU run once the container starts) is still
  unverified end-to-end.
