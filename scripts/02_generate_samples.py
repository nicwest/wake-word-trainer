#!/usr/bin/env python3
"""Generates raw (unaugmented) positive TTS samples for a wake word phrase
using piper-sample-generator's LibriTTS-R generator checkpoint. Augmentation
(reverb, background noise, gain, etc.) happens later in 03_build_features.py
via microwakeword's own Augmentation class, not here.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mww_trainer import paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wake_word", help='Phrase to synthesize, e.g. "hey wild rider"')
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=50, help="Raise this on a GPU; 1-2 on CPU")
    parser.add_argument("--max-speakers", type=int, default=800, help="LibriTTS-R has 904 speakers; the last few are noisy")
    parser.add_argument(
        "--generator",
        default=None,
        help="Path to the .pt generator checkpoint (default: assets dir from 01_fetch_assets.py)",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if output-dir already has samples")
    args = parser.parse_args()

    generator = Path(args.generator) if args.generator else paths.piper_dir() / "en_US-libritts_r-medium.pt"
    generator_config = generator.with_name(generator.name + ".json")
    if not generator.exists():
        sys.exit(f"Generator checkpoint not found at {generator}. Run 01_fetch_assets.py first.")
    if not generator_config.exists():
        sys.exit(f"Generator config not found at {generator_config}. Run 01_fetch_assets.py (--force to redownload) first.")

    src_dir = paths.piper_sample_generator_src_dir()
    if not src_dir.exists():
        sys.exit(f"piper-sample-generator source not found at {src_dir}. Run 01_fetch_assets.py first.")

    out_dir = paths.generated_samples_dir(args.wake_word)
    if out_dir.exists() and any(out_dir.glob("*.wav")) and not args.force:
        print(f"{out_dir} already has samples; pass --force to regenerate.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "piper_sample_generator",
        args.wake_word,
        "--model",
        str(generator),
        "--max-samples",
        str(args.max_samples),
        "--batch-size",
        str(args.batch_size),
        "--max-speakers",
        str(args.max_speakers),
        "--output-dir",
        str(out_dir),
    ]
    # Run against the git-cloned source (src_dir) rather than a pip-installed
    # package: piper_sample_generator's __main__.py imports its sibling
    # piper_train package, which PyPI never bundles (see requirements.txt).
    # Prepending src_dir to PYTHONPATH makes both importable from the clone.
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"Wrote samples to {out_dir}")


if __name__ == "__main__":
    main()
