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


def features_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "features"


def real_features_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "real_features"


def training_config_path(wake_word: str) -> Path:
    return work_dir(wake_word) / "training_parameters.yaml"


def train_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "trained_models"


def output_dir(wake_word: str) -> Path:
    return work_dir(wake_word) / "output"
