#!/usr/bin/env python3
"""Augments the generated positive samples and extracts spectrogram features
into RaggedMmap train/validation/testing sets. Mirrors the reference
microWakeWord notebook's augmentation + feature-generation cells.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mww_trainer import paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wake_word")
    parser.add_argument("--augmentation-duration-s", type=float, default=3.2)
    parser.add_argument("--background-min-snr-db", type=int, default=-5)
    parser.add_argument("--background-max-snr-db", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from microwakeword.audio.augmentation import Augmentation
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap

    samples_dir = paths.generated_samples_dir(args.wake_word)
    if not samples_dir.exists() or not any(samples_dir.glob("*.wav")):
        sys.exit(f"No generated samples in {samples_dir}. Run 02_generate_samples.py first.")

    impulse_paths = [str(paths.rir_dir())] if paths.rir_dir().exists() else []
    background_paths = [
        str(p) for p in (paths.background_dir() / "audioset", paths.background_dir() / "fma") if p.exists()
    ]
    if not impulse_paths:
        print("WARNING: no RIR assets found, training without reverb augmentation.")
    if not background_paths:
        print("WARNING: no background-noise assets found, training without background-noise augmentation.")

    clips = Clips(
        input_directory=str(samples_dir),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=False,
        random_split_seed=10,
        split_count=0.1,
    )
    augmenter = Augmentation(
        augmentation_duration_s=args.augmentation_duration_s,
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
        background_min_snr_db=args.background_min_snr_db,
        background_max_snr_db=args.background_max_snr_db,
        min_jitter_s=0.195,
        max_jitter_s=0.205,
    )

    out_root = paths.features_dir(args.wake_word)
    splits = ["training", "validation", "testing"]
    for split in splits:
        out_dir = out_root / split
        mmap_dir = out_dir / "wakeword_mmap"
        if mmap_dir.exists() and not args.force:
            print(f"{mmap_dir} already exists; pass --force to rebuild.")
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

        print(f"[{split}] generating features -> {mmap_dir}")
        RaggedMmap.from_generator(
            out_dir=str(mmap_dir),
            sample_generator=spectrograms.spectrogram_generator(split=split_name, repeat=repetition),
            batch_size=100,
            verbose=True,
        )

    print(f"Features ready at {out_root}")


if __name__ == "__main__":
    main()
