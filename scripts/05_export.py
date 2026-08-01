#!/usr/bin/env python3
"""Picks the best-calibrated cutoff from training's streaming ROC results,
copies the quantized streaming tflite model, and writes an ESPHome
micro_wake_word-compatible manifest JSON (same schema as an existing
hey_wild_rider.json / okay_nabu.json model manifest).
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mww_trainer import paths  # noqa: E402

ROC_LINE_RE = re.compile(r"Cutoff (?P<cutoff>[\d.]+): frr=(?P<frr>[\d.]+); faph=(?P<faph>[\d.]+)")


def parse_roc(roc_path: Path):
    points = []
    for line in roc_path.read_text().splitlines():
        m = ROC_LINE_RE.match(line.strip())
        if m:
            points.append(
                {"cutoff": float(m["cutoff"]), "frr": float(m["frr"]), "faph": float(m["faph"])}
            )
    return points


def pick_cutoff(points, max_faph: float) -> dict:
    viable = [p for p in points if p["faph"] <= max_faph]
    if viable:
        # Lowest false-rejection-rate (best recall) among candidates that meet
        # the ambient false-accept budget.
        return min(viable, key=lambda p: p["frr"])
    # Nothing met the budget: fall back to whichever cutoff had the fewest
    # false accepts per hour, even if that's not great.
    print(f"WARNING: no cutoff kept false-accepts/hour <= {max_faph}; falling back to lowest-faph candidate.")
    return min(points, key=lambda p: p["faph"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wake_word")
    parser.add_argument("--author", default="")
    parser.add_argument("--website", default="")
    parser.add_argument("--max-faph", type=float, default=2.0, help="Max acceptable ambient false-accepts/hour when picking the cutoff")
    parser.add_argument(
        "--tensor-arena-size",
        type=int,
        default=32000,
        help="Not computed from the model -- a starting guess. ESPHome will refuse to load the model if this is too "
        "small; bump it and re-flash if you see an arena allocation failure.",
    )
    parser.add_argument("--sliding-window-size", type=int, default=5)
    parser.add_argument("--minimum-esphome-version", default="2024.7.0")
    args = parser.parse_args()

    tflite_run_dir = paths.train_dir(args.wake_word) / "tflite_stream_state_internal_quant"
    tflite_src = tflite_run_dir / "stream_state_internal_quant.tflite"
    roc_path = tflite_run_dir / "tflite_streaming_roc.txt"
    if not tflite_src.exists():
        sys.exit(f"No trained model at {tflite_src}. Run 04_train.py first.")

    slug = paths.slugify(args.wake_word)
    out_dir = paths.output_dir(args.wake_word)
    out_dir.mkdir(parents=True, exist_ok=True)
    tflite_dest = out_dir / f"{slug}.tflite"
    shutil.copy2(tflite_src, tflite_dest)

    probability_cutoff = 0.97  # microWakeWord's own conservative fallback
    if roc_path.exists():
        points = parse_roc(roc_path)
        if points:
            chosen = pick_cutoff(points, args.max_faph)
            probability_cutoff = chosen["cutoff"]
            print(
                f"Chosen cutoff {probability_cutoff:.2f} "
                f"(frr={chosen['frr']:.4f}, faph={chosen['faph']:.3f}, budget<= {args.max_faph}/hr)"
            )
        else:
            print(f"WARNING: could not parse any cutoffs from {roc_path}; using default {probability_cutoff}")
    else:
        print(f"WARNING: {roc_path} not found; using default cutoff {probability_cutoff}. Re-run training with --test_tflite_streaming_quantized 1.")

    manifest = {
        "type": "micro",
        "wake_word": args.wake_word,
        "author": args.author,
        "website": args.website,
        "model": f"{slug}.tflite",
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": round(probability_cutoff, 2),
            "feature_step_size": 10,
            "sliding_window_size": args.sliding_window_size,
            "tensor_arena_size": args.tensor_arena_size,
            "minimum_esphome_version": args.minimum_esphome_version,
        },
    }
    manifest_path = out_dir / f"{slug}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {tflite_dest}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
