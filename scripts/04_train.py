#!/usr/bin/env python3
"""Writes training_parameters.yaml and runs microwakeword.model_train_eval.

Feature-set weights and the mixednet architecture args mirror the reference
microWakeWord notebook's defaults -- change them here (or via the CLI flags
below) as you tune for your specific wake word.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mww_trainer import paths  # noqa: E402

NEGATIVE_FEATURE_WEIGHTS = {
    "speech": {"sampling_weight": 10.0, "penalty_weight": 1.0, "truncation_strategy": "random"},
    "dinner_party": {"sampling_weight": 10.0, "penalty_weight": 1.0, "truncation_strategy": "random"},
    "no_speech": {"sampling_weight": 5.0, "penalty_weight": 1.0, "truncation_strategy": "random"},
    # Eval-only: zero sampling weight means it's never trained on, only used
    # for the ambient false-accept-per-hour validation/test metrics.
    "dinner_party_eval": {"sampling_weight": 0.0, "penalty_weight": 1.0, "truncation_strategy": "split"},
}


def build_config(
    wake_word: str,
    training_steps: int,
    real_sampling_weight: float,
    real_penalty_weight: float,
    false_positive_sampling_weight: float,
    false_positive_penalty_weight: float,
) -> dict:
    # Two different dials, easy to conflate:
    #   sampling_weight -- how often a set fills a batch slot (a mix ratio).
    #     High values on a SMALL set (real/false_positive, tens of clips vs.
    #     thousands in the shared negative sets) mean the model rehearses
    #     that same narrow pool over and over relative to everything else --
    #     that's overfitting-to-a-few-clips, not "generalizing better", and
    #     is what regressed recall when false_positive_sampling_weight was
    #     pushed to 15 (see git history). Keep these modest, roughly
    #     reflecting how much real diversity each set actually has.
    #   penalty_weight -- how much a MISTAKE on that set costs in the loss,
    #     independent of how often it's sampled. This is the more direct way
    #     to say "care more about getting these specific examples right"
    #     without forcing more repetition. (There's also a much bigger
    #     structural asymmetry already baked in below: negative_class_weight
    #     20 vs positive_class_weight 1, applied globally regardless of
    #     which set an example came from.)
    features = [
        {
            "features_dir": str(paths.features_dir(wake_word)),
            "sampling_weight": 2.0,
            "penalty_weight": 1.0,
            "truth": True,
            "truncation_strategy": "truncate_start",
            "type": "mmap",
        }
    ]

    real_features_dir = paths.real_features_dir(wake_word)
    if real_features_dir.exists() and any(real_features_dir.glob("*/wakeword_mmap")):
        # Real (non-synthetic) positives -- e.g. promoted false-negative
        # captures -- kept as their own weighted feature set rather than
        # merged into the TTS set, so they can be weighted differently:
        # there are usually far fewer of them, but they're the most
        # representative signal for real-world recall.
        features.append(
            {
                "features_dir": str(real_features_dir),
                "sampling_weight": real_sampling_weight,
                "penalty_weight": real_penalty_weight,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        )
    else:
        print(f"No real_samples features at {real_features_dir} (run 03_build_features.py after adding real_samples/ clips, if you have any).")

    false_positive_features_dir = paths.false_positive_features_dir(wake_word)
    if false_positive_features_dir.exists() and any(false_positive_features_dir.glob("*/wakeword_mmap")):
        # Captured false wakes -- negative examples, and the highest-value
        # kind: hand-picked instances of exactly what this model gets wrong,
        # not generic negative audio. penalty_weight up (mistakes on these
        # cost more), sampling_weight kept modest (this set is tiny --
        # rehearsing it too often is how we regressed recall last time).
        features.append(
            {
                "features_dir": str(false_positive_features_dir),
                "sampling_weight": false_positive_sampling_weight,
                "penalty_weight": false_positive_penalty_weight,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            }
        )
    else:
        print(f"No false_positive_samples features at {false_positive_features_dir} (run 03_build_features.py after adding false_positive_samples/ clips, if you have any).")

    for name, weights in NEGATIVE_FEATURE_WEIGHTS.items():
        feature_dir = paths.negative_features_dir() / name
        if not feature_dir.exists():
            print(f"WARNING: {feature_dir} missing, skipping. Run 01_fetch_assets.py.")
            continue
        features.append(
            {
                "features_dir": str(feature_dir),
                "sampling_weight": weights["sampling_weight"],
                "penalty_weight": weights["penalty_weight"],
                "truth": False,
                "truncation_strategy": weights["truncation_strategy"],
                "type": "mmap",
            }
        )

    return {
        "window_step_ms": 10,
        "train_dir": str(paths.train_dir(wake_word)),
        "features": features,
        "training_steps": [training_steps],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wake_word")
    parser.add_argument("--training-steps", type=int, default=10000)
    parser.add_argument("--restore-checkpoint", type=int, default=1, help="Resume from checkpoint if the train dir already has one")
    parser.add_argument("--real-sampling-weight", type=float, default=4.0, help="Sampling weight for real_samples/ features, if present (higher than generated's 2.0 -- there are usually far fewer of them)")
    parser.add_argument("--real-penalty-weight", type=float, default=2.0, help="Loss penalty weight for real_samples/ mistakes, if present")
    parser.add_argument("--false-positive-sampling-weight", type=float, default=4.0, help="Sampling weight for false_positive_samples/ features, if present. Keep modest -- this set is small (tens of clips), and a high value means the model rehearses it disproportionately rather than generalizing (regressed recall at 15.0, see git history)")
    parser.add_argument("--false-positive-penalty-weight", type=float, default=3.0, help="Loss penalty weight for false_positive_samples/ mistakes, if present -- the more direct 'care about these' dial, without the overfitting risk of a high sampling weight")
    parser.add_argument("--config-only", action="store_true", help="Only write the yaml, don't launch training")
    args = parser.parse_args()

    config = build_config(
        args.wake_word,
        args.training_steps,
        args.real_sampling_weight,
        args.real_penalty_weight,
        args.false_positive_sampling_weight,
        args.false_positive_penalty_weight,
    )
    config_path = paths.training_config_path(args.wake_word)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Wrote {config_path}")

    if args.config_only:
        return

    cmd = [
        sys.executable,
        "-m",
        "microwakeword.model_train_eval",
        f"--training_config={config_path}",
        "--train",
        "1",
        "--restore_checkpoint",
        str(args.restore_checkpoint),
        "--test_tf_nonstreaming",
        "0",
        "--test_tflite_nonstreaming",
        "0",
        "--test_tflite_nonstreaming_quantized",
        "0",
        "--test_tflite_streaming",
        "0",
        "--test_tflite_streaming_quantized",
        "1",
        "--use_weights",
        "best_weights",
        "mixednet",
        "--pointwise_filters",
        "64,64,64,64",
        "--repeat_in_block",
        "1, 1, 1, 1",
        "--mixconv_kernel_sizes",
        "[5], [7,11], [9,15], [23]",
        "--residual_connection",
        "0,0,0,0",
        "--first_conv_filters",
        "32",
        "--first_conv_kernel_size",
        "5",
        "--stride",
        "3",
    ]
    # microwakeword isn't pip/uv installed -- see the note in requirements.txt.
    # Put the git clone on PYTHONPATH so `-m microwakeword.model_train_eval`
    # can find it (and its audio/ subpackage) in this subprocess.
    if not paths.microwakeword_src_dir().exists():
        sys.exit(f"microwakeword source not found at {paths.microwakeword_src_dir()}. Run 01_fetch_assets.py first.")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(paths.microwakeword_src_dir()) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
