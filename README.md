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

Everything that should survive between sessions -- the venv, downloaded
assets, and trained models -- lives under one directory, `$MWW_DATA_DIR`.
Point that at a RunPod persistent Network Volume and a fresh pod skips
straight to training with no re-bootstrap: the install (via
[uv](https://astral.sh/uv), already present on `runpod/base` images) only
happens once *per volume*, and every pod after that reattaches the same
volume regardless of which host it lands on.

(We tried baking `requirements.txt` into a custom Docker image instead, so a
pod wouldn't need to install anything at all. Reverted -- pulling that image
on a RunPod host that hadn't cached it was just as slow as the install it was
supposed to avoid, and unlike a persistent volume, a pulled image doesn't
follow you to a different host next time. The venv-on-volume approach below
is the one-time cost that actually stays paid.)

## RunPod setup

Pick a plain, small pod image -- no need for anything ML-specific, since the
venv brings its own CUDA runtime (see "Disk footprint" below for why). RunPod's
own `runpod/base:1.1.0-ubuntu2204` works well and pulls in seconds even on a
fresh host. Attach a persistent Network Volume at `/workspace/data`, and set
the pod's container start command to clone this repo and run it:

```sh
git clone https://github.com/nicwest/wake-word-trainer.git /app && \
cd /app && \
MWW_DATA_DIR=/workspace/data ./train.sh "hey wild rider"
```

First run on a fresh volume pays the full install + asset download cost.
Every subsequent pod that reattaches the same volume (even on a different
host) skips both entirely and goes straight to sample generation.

## Layout

```
$MWW_DATA_DIR/
  venv/                        one combined venv (torch + tensorflow), managed by uv; reused if requirements.txt is unchanged
  assets/                      shared across every wake word you train
    piper/                     LibriTTS-R generator checkpoint (~200MB)
    piper-sample-generator-src/  git clone, not pip/uv installed -- see requirements.txt
    microwakeword-src/          git clone, not pip/uv installed -- see requirements.txt
    rir/                       MIT environmental impulse responses (~8MB)
    background/                curated background-noise subset (audioset clips + fma_xs)
    negative_features/         pre-extracted negative spectrogram sets (speech, dinner_party, dinner_party_eval, no_speech)
  work/<wake_word_slug>/       one subtree per phrase you train
    generated_samples/         raw TTS positives
    real_samples/               real (non-synthetic) positives you drop in yourself -- see below
    false_positive_samples/     captured false wakes (negative examples) -- see below
    features/{training,validation,testing}/                features for generated_samples/
    real_features/{training,validation,testing}/            features for real_samples/, if any
    false_positive_features/{training,validation,testing}/  features for false_positive_samples/, if any
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

### Captured false positives (false wakes)

The mirror-image case: clips where the model triggered but the wake word
wasn't actually said. These are **negative** examples -- the same idea as
Tater's "mark as False wake" review flow, hand-picked instances of exactly
what this model gets wrong, which makes them the highest-value negative
signal available.

Drop them in `work/<wake_word_slug>/false_positive_samples/` and re-run
`03_build_features.py` -- same format rules as `real_samples/`. They get
their own feature pipeline (`false_positive_features/`) and their own
`training_parameters.yaml` entry with `truth: false`. Skipped silently if
empty, same as `real_samples/`.

**`sampling_weight` vs `penalty_weight`** -- easy to conflate, and worth
understanding before tuning either:
- `sampling_weight` controls how often a set fills a training-batch slot (a
  mix ratio across all feature sets). `real_samples/`/`false_positive_samples/`
  are typically tens of clips, tiny next to the shared `negative_features/`
  sets (thousands). Pushing sampling_weight high on a small set doesn't
  teach general robustness -- it makes the model rehearse that narrow pool
  disproportionately, which looks like overfitting. Confirmed this
  regressing recall in practice at `false_positive_sampling_weight=15`
  (default is `4.0`, comparable to `no_speech`'s `5.0` -- see git history).
- `penalty_weight` controls how much a *mistake* on that set costs in the
  loss, independent of sampling frequency -- the more direct "care about
  these specifically" dial. Defaults: `real_samples/` at `2.0`,
  `false_positive_samples/` at `3.0` (both `1.0` elsewhere), via
  `04_train.py --real-penalty-weight`/`--false-positive-penalty-weight`.
- Underneath both, `negative_class_weight: 20` vs `positive_class_weight: 1`
  already penalizes *any* negative-class mistake 20x more than a positive
  one, globally, regardless of which set it came from -- the dominant force
  in the whole config, bigger than any individual set's weights.

## Quick start

```sh
export MWW_DATA_DIR=/workspace/data   # wherever your persistent volume is mounted (./data for local use)
./train.sh "hey wild rider"
```

Re-running the same command on a fresh pod (same `$MWW_DATA_DIR`) skips venv
install and asset downloads entirely and goes straight to sample generation.

Individual steps (useful for iterating on one wake word without re-touching
shared assets):

```sh
source scripts/00_setup_venv.sh
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
actual URLs this harness downloads (checked 2026-08-01), not from memory:

| Component | On-disk size |
|---|---|
| `venv/` (torch 2.9.x + tensorflow[and-cuda] + shared cu12 CUDA libs) | ~5-6 GB |
| `assets/piper/` (LibriTTS-R generator checkpoint) | ~0.2 GB |
| `assets/rir/` (MIT impulse responses, already 16kHz) | ~0.01 GB |
| `assets/background/audioset/` (2000 clips, default) | ~0.6 GB |
| `assets/background/fma/` (fma_xs, converted to 16kHz PCM) | ~0.3-0.4 GB |
| `assets/negative_features/` (4 extracted zips, ~5.7GB compressed) | ~6-7 GB |
| `work/<wake_word>/` (samples + features + checkpoints + output) | ~0.3-0.8 GB per phrase |

**Total for the shared assets + venv + one wake word: roughly 13-15 GB.**
Give the volume real headroom beyond that: extraction briefly needs the zip
*and* its extracted contents on disk at once (the negative-features step
peaks around 12GB transiently before cleanup), and TF checkpoints accumulate
during a run. **Size the RunPod volume at 30-40GB**, not 15GB.

The `nvidia-*-cu12` pin in `requirements.txt` matters for the venv number:
letting torch resolve to its latest version pulls `nvidia-*-cu13` packages
instead, which don't overlap with tensorflow's `cu12` requirement -- that
would install two full CUDA library sets side by side (~2-3GB extra) instead
of one shared set. It's also why the pod image itself doesn't need to be
CUDA-flavored at all -- these packages bring their own CUDA runtime, the
image just needs Python (and uv).

## Alternative: bucket sync instead of a Network Volume

Network Volumes lock you to one RunPod datacenter (and one pod at a time),
which limits which GPUs you can rent. If that matters more than the
convenience, sync `$MWW_DATA_DIR` to a bucket instead: `rclone sync` it to
R2/B2/S3 at the start and end of each pod session. That's pod-to-bucket only
(datacenter-speed transfer, not your home connection), but it does mean every
session pays a sync delay proportional to `$MWW_DATA_DIR`'s size -- worth it
only if GPU availability is the bigger constraint.

## Known gaps / next steps

- `05_export.py --tensor-arena-size` is a starting guess (32000), not
  computed from the model. ESPHome will fail to load the model if it's too
  small -- bump it and re-flash if that happens.
- No adversarial/similar-phrase negative generation (openWakeWord does this
  in the full microWakeWord recipe). Worth adding if false-accepts on
  phonetically similar words are a problem.
- The pip -> uv migration (requirements.txt install path, and moving
  microwakeword from an editable git install to a plain clone + sys.path/
  PYTHONPATH injection, since uv has no editable-from-a-Git-URL support at
  all) has only been verified locally on macOS with a throwaway lightweight
  requirements.txt -- not yet run for real on a pod with the actual
  torch/tensorflow/microwakeword stack. Treat the next RunPod run as the
  real test of that specific change.

The core pipeline itself (fetch -> generate -> build_features -> train ->
export) has completed successfully end to end multiple times at this point,
producing real, usable `.tflite` + manifest output -- see the project's
commit history for the specific bugs that got found and fixed along the way
(dead upstream URLs, missing upstream `__init__.py`, dependency collisions,
etc.). Most of the debugging churn has been in one-time environment setup,
not the training logic itself.
