#!/usr/bin/env python3
"""Fetches the curated, size-minimized set of shared training assets into
$MWW_DATA_DIR/assets. Everything here is wake-word-independent and downloaded
once; re-runs are no-ops unless --force is passed.

Deliberately smaller than the reference recipes:
  - RIR: the full MIT_environmental_impulse_responses set (270 short clips,
    ~8MB) instead of the much larger BIRD RIR corpus.
  - Background noise: a capped number of clips streamed from AudioSet's
    "balanced" config (the full balanced-train split is ~26GB across 38
    parquet shards; we only pull --audioset-clips of them) + the FMA "extra
    small" subset (~180MB), instead of the full AudioSet/FMA/WHAM/CHiME
    battery. Trades some robustness breadth for a much smaller download;
    raise --audioset-clips or add more background_paths later if
    false-accepts are too high.
  - Negative features: the same 4 pre-extracted spectrogram feature sets
    microWakeWord's author publishes (speech, dinner_party, dinner_party_eval,
    no_speech). These are already-augmented spectrograms, not raw audio, and
    there's no smaller substitute without giving up false-accept robustness.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mww_trainer import paths  # noqa: E402

PIPER_GENERATOR_URL = "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"

# Streamed via the `datasets` library rather than a hardcoded file URL: the
# upstream repo has reorganized its raw file layout before (tar shards ->
# per-split parquet files) and streaming through the loader script survives
# that kind of change.
AUDIOSET_DATASET_NAME = "agkphysics/AudioSet"
AUDIOSET_CONFIG = "balanced"
DEFAULT_AUDIOSET_CLIPS = 2000  # ~5.5 hours of 10s clips, roughly one parquet shard's worth

FMA_XS_URL = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"

NEGATIVE_FEATURES_URL_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main"
NEGATIVE_FEATURE_SETS = ["speech", "dinner_party", "dinner_party_eval", "no_speech"]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def marker(dir_: Path) -> Path:
    return dir_ / ".done"


def is_done(dir_: Path) -> bool:
    return marker(dir_).exists()


def mark_done(dir_: Path) -> None:
    marker(dir_).touch()


def fetch_piper_generator(force: bool) -> None:
    out_dir = paths.piper_dir()
    dest = out_dir / "en_US-libritts_r-medium.pt"
    if dest.exists() and not force:
        print(f"[piper] already have {dest}")
        return
    print("[piper] downloading generator checkpoint (~75 MB)")
    download(PIPER_GENERATOR_URL, dest)


def fetch_rir(force: bool) -> None:
    out_dir = paths.rir_dir()
    if is_done(out_dir) and not force:
        print(f"[rir] already have {out_dir}")
        return
    print("[rir] downloading MIT environmental impulse responses")
    import numpy as np
    import scipy.io.wavfile
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    )
    count = 0
    for row in rir_dataset:
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            str(out_dir / name), 16000, (row["audio"]["array"] * 32767).astype(np.int16)
        )
        count += 1
    print(f"[rir] wrote {count} impulse responses")
    mark_done(out_dir)


def _convert_dir_to_16k_wav(src_glob_dir: Path, src_pattern: str, out_dir: Path) -> int:
    import numpy as np
    import scipy.io.wavfile
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    files = [str(p) for p in src_glob_dir.glob(src_pattern)]
    if not files:
        return 0
    ds = datasets.Dataset.from_dict({"audio": files})
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    count = 0
    for row in ds:
        name = Path(row["audio"]["path"]).stem + ".wav"
        scipy.io.wavfile.write(
            str(out_dir / name), 16000, (row["audio"]["array"] * 32767).astype(np.int16)
        )
        count += 1
    return count


def fetch_audioset(max_clips: int, force: bool) -> None:
    out_dir = paths.background_dir() / "audioset"
    if is_done(out_dir) and not force:
        print(f"[audioset] already have {out_dir}")
        return

    print(f"[audioset] streaming up to {max_clips} clips from {AUDIOSET_DATASET_NAME} ({AUDIOSET_CONFIG})")
    import itertools

    import numpy as np
    import scipy.io.wavfile
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    audioset = datasets.load_dataset(
        AUDIOSET_DATASET_NAME, AUDIOSET_CONFIG, split="train", streaming=True
    )
    audioset = audioset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    count = 0
    for row in itertools.islice(audioset, max_clips):
        name = f"{row['video_id']}.wav"
        scipy.io.wavfile.write(
            str(out_dir / name), 16000, (row["audio"]["array"] * 32767).astype(np.int16)
        )
        count += 1
    print(f"[audioset] wrote {count} clips")
    mark_done(out_dir)


def fetch_fma(force: bool, keep_downloads: bool) -> None:
    out_dir = paths.background_dir() / "fma"
    if is_done(out_dir) and not force:
        print(f"[fma] already have {out_dir}")
        return

    download_dir = paths.assets_dir() / "downloads" / "fma"
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "fma_xs.zip"
    if not zip_path.exists():
        print("[fma] downloading fma_xs.zip")
        download(FMA_XS_URL, zip_path)

    raw_dir = download_dir / "raw"
    if not raw_dir.exists():
        print("[fma] extracting fma_xs.zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    print("[fma] converting to 16kHz wav")
    count = _convert_dir_to_16k_wav(raw_dir, "**/*.mp3", out_dir)
    print(f"[fma] wrote {count} clips")
    mark_done(out_dir)

    # The zip + extracted mp3s are pure intermediate scratch once converted;
    # they roughly double disk use if left around, for no benefit.
    if not keep_downloads:
        import shutil

        shutil.rmtree(download_dir, ignore_errors=True)


def fetch_negative_features(force: bool, keep_downloads: bool) -> None:
    out_dir = paths.negative_features_dir()
    downloads_dir = paths.assets_dir() / "downloads" / "negative_features"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in NEGATIVE_FEATURE_SETS:
        set_dir = out_dir / name
        if is_done(set_dir) and not force:
            print(f"[negative_features] already have {name}")
            continue
        zip_path = downloads_dir / f"{name}.zip"
        if not zip_path.exists():
            print(f"[negative_features] downloading {name}.zip")
            download(f"{NEGATIVE_FEATURES_URL_ROOT}/{name}.zip", zip_path)
        print(f"[negative_features] extracting {name}.zip")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        mark_done(set_dir)
        if not keep_downloads:
            zip_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download/rebuild even if present")
    parser.add_argument(
        "--skip-background",
        action="store_true",
        help="Skip AudioSet/FMA background noise (only useful for a fast smoke test; hurts model quality)",
    )
    parser.add_argument(
        "--audioset-clips",
        type=int,
        default=DEFAULT_AUDIOSET_CLIPS,
        help="How many AudioSet balanced-train clips to stream down (10s clips each)",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep intermediate zip/mp3 downloads after extracting (default: delete them, they roughly double disk use for no benefit once the final wav/feature files exist)",
    )
    args = parser.parse_args()

    fetch_piper_generator(args.force)
    fetch_rir(args.force)
    if not args.skip_background:
        fetch_audioset(args.audioset_clips, args.force)
        fetch_fma(args.force, args.keep_downloads)
    fetch_negative_features(args.force, args.keep_downloads)

    print("Assets ready at", paths.assets_dir())


if __name__ == "__main__":
    main()
