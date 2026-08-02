"""Single-directory layout for everything that should survive between RunPod sessions.

Point MWW_DATA_DIR at a persistent volume (or a bucket-synced local mount) and
every asset, venv, and trained model lands under it. Nothing outside this tree
is state -- the rest of the repo is just code.
"""

import os
import re
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("MWW_DATA_DIR", "./data")).resolve()


def venv_dir() -> Path:
    return data_dir() / "venv"


def assets_dir() -> Path:
    return data_dir() / "assets"


def piper_dir() -> Path:
    return assets_dir() / "piper"


def piper_sample_generator_src_dir() -> Path:
    """Git clone of the piper-sample-generator repo (not the PyPI package --
    see the note in requirements.txt for why). Put on PYTHONPATH when
    invoking `python -m piper_sample_generator` so the sibling piper_train
    package it imports is importable too.
    """
    return assets_dir() / "piper-sample-generator-src"


def microwakeword_src_dir() -> Path:
    """Git clone of the microWakeWord repo, pinned to a specific commit --
    not pip/uv installed. microwakeword/audio/ has no __init__.py upstream,
    and `find_packages()` (in setup.py) silently drops that whole subpackage
    from a real wheel build; uv (unlike pip) has no editable-install-from-a-
    Git-URL escape hatch to work around it the way requirements.txt used to.
    Same fix as piper_sample_generator_src_dir() above: clone it ourselves
    and put it on sys.path/PYTHONPATH wherever it's imported instead.
    """
    return assets_dir() / "microwakeword-src"


def rir_dir() -> Path:
    return assets_dir() / "rir"


def background_dir() -> Path:
    return assets_dir() / "background"


def negative_features_dir() -> Path:
    return assets_dir() / "negative_features"


def slugify(wake_word: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", wake_word.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"Could not derive a slug from wake word {wake_word!r}")
    return slug


def work_dir(wake_word: str) -> Path:
    return data_dir() / "work" / slugify(wake_word)


def generated_samples_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "generated_samples"


def real_samples_dir(wake_word: str) -> Path:
    """Real (non-synthetic) positive recordings -- e.g. device-captured false
    negatives you're promoting into training data. Drop 16kHz-or-not
    wav/flac/mp3/ogg files in here; kept separate from generated_samples/ so
    they can be weighted differently in training_parameters.yaml instead of
    getting diluted into the much larger TTS-generated set.
    """
    return work_dir(wake_word) / "real_samples"


def false_positive_samples_dir(wake_word: str) -> Path:
    """Captured false wakes -- clips where the model (or a predecessor of it)
    triggered but the wake word wasn't actually said. These are negative
    examples, kept as their own weighted feature set (like real_samples/ is
    for positives) rather than merged into the shared negative_features/
    assets, since they're hand-picked examples of exactly what this specific
    model gets wrong.
    """
    return work_dir(wake_word) / "false_positive_samples"


def features_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "features"


def real_features_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "real_features"


def false_positive_features_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "false_positive_features"


def training_config_path(wake_word: str) -> Path:
    return work_dir(wake_word) / "training_parameters.yaml"


def train_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "trained_models"


def output_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "output"
