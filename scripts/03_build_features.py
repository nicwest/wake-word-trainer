#!/usr/bin/env python3
"""Augments positive samples and extracts spectrogram features into RaggedMmap
train/validation/testing sets. Mirrors the reference microWakeWord notebook's
augmentation + feature-generation cells.

Runs over three independent sources, each producing its own feature set so
04_train.py can weight them separately:
  - generated_samples/ -- synthetic TTS positives from 02_generate_samples.py
    (required)
  - real_samples/ -- real, non-synthetic positives you drop in yourself (e.g.
    device-captured false negatives you're promoting into training data;
    optional, skipped with a note if empty)
  - false_positive_samples/ -- captured false wakes: clips where the model
    triggered but the wake word wasn't said. Negative examples, not positive
    (optional, skipped with a note if empty)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mww_trainer import paths  # noqa: E402

# Not pip/uv installed -- see the note in requirements.txt and
# mww_trainer.paths.microwakeword_src_dir's docstring for why.
sys.path.insert(0, str(paths.microwakeword_src_dir()))


def build_augmenter(augmentation_duration_s, background_min_snr_db, background_max_snr_db):
    from microwakeword.audio.augmentation import Augmentation

    impulse_paths = [str(paths.rir_dir())] if paths.rir_dir().exists() else []
    background_paths = [
        str(p) for p in (paths.background_dir() / "audioset", paths.background_dir() / "fma") if p.exists()
    ]
    if not impulse_paths:
        print("WARNING: no RIR assets found, training without reverb augmentation.")
    if not background_paths:
        print("WARNING: no background-noise assets found, training without background-noise augmentation.")

    return Augmentation(
        augmentation_duration_s=augmentation_duration_s,
        augmentation_probabilities={
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.1,
            "PitchShift": 0.1,
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.75 if background_paths else 0.0,
            "Gain": 1.0,
            "RIR": 0.5 if impulse_paths else 0.0,
        },
        impulse_paths=impulse_paths,
        background_paths=background_paths,
        background_min_snr_db=background_min_snr_db,
        background_max_snr_db=background_max_snr_db,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )


def build_features(label: str, samples_dir: Path, out_root: Path, augmenter, force: bool) -> None:
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    sample_count = len(list(samples_dir.glob("*.wav"))) if samples_dir.exists() else 0
    if sample_count == 0:
        print(f"[{label}] no samples in {samples_dir}, skipping.")
        return

    clips = Clips(
        input_directory=str(samples_dir),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )

    splits = ["training", "validation", "testing"]
    for split in splits:
        out_dir = out_root / split
        mmap_dir = out_dir / "wakeword_mmap"
        if mmap_dir.exists() and not force:
            print(f"[{label}] {mmap_dir} already exists; pass --force to rebuild.")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        if split == "testing":
            # Streaming-mode test set: no artificial repetition/frame-sliding.
            split_name, repetition = "test", 1
            spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10)
        else:
            # Non-streaming train/validation: slide over the spectrogram to
            # simulate streaming inference without regenerating audio.
            split_name = "validation" if split == "validation" else "train"
            repetition = 1 if split == "validation" else 2
            spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10)

        print(f"[{label}/{split}] generating features from {sample_count} clips -> {mmap_dir}")
        try:
            RaggedMmap.from_generator(
                out_dir=str(mmap_dir),
                sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
                batch_size=100,
                verbose=True,
            )
        except (ValueError, IndexError) as e:
            # HF's train_test_split needs enough clips to carve out a
            # validation+test slice; a handful of real_samples/ clips can hit
            # this before you've collected enough to be worth splitting.
            sys.exit(
                f"[{label}] failed building the '{split}' split ({e}). "
                f"With only {sample_count} clips there may not be enough to split "
                "into train/validation/testing -- add more clips, or lower --split-count."
            )

    print(f"[{label}] features ready at {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wake_word")
    parser.add_argument("--augmentation-duration-s", type=float, default=3.2)
    parser.add_argument("--background-min-snr-db", type=int, default=-5)
    parser.add_argument("--background-max-snr-db", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Force-rebuild all three feature sets, including the large generated_samples one")
    parser.add_argument(
        "--force-uploaded",
        action="store_true",
        help="Force-rebuild only real_samples/false_positive_samples features (generated_samples still cached unless --force too) -- use after uploading new captures with upload_samples.sh",
    )
    args = parser.parse_args()

    if not paths.microwakeword_src_dir().exists():
        sys.exit(f"microwakeword source not found at {paths.microwakeword_src_dir()}. Run 01_fetch_assets.py first.")

    generated_dir = paths.generated_samples_dir(args.wake_word)
    if not generated_dir.exists() or not any(generated_dir.glob("*.wav")):
        sys.exit(f"No generated samples in {generated_dir}. Run 02_generate_samples.py first.")

    augmenter = build_augmenter(args.augmentation_duration_s, args.background_min_snr_db, args.background_max_snr_db)
    force_uploaded = args.force or args.force_uploaded

    build_features("generated", generated_dir, paths.features_dir(args.wake_word), augmenter, args.force)
    build_features("real", paths.real_samples_dir(args.wake_word), paths.real_features_dir(args.wake_word), augmenter, force_uploaded)
    build_features(
        "false_positive",
        paths.false_positive_samples_dir(args.wake_word),
        paths.false_positive_features_dir(args.wake_word),
        augmenter,
        force_uploaded,
    )


if __name__ == "__main__":
    main()
