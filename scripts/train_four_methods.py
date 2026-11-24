"""Train the four LoRA Lens PEFT recipes and write a comparison bundle.

Does not overwrite results/four_methods_bundle/ unless you point --output there.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs import FOUR_METHODS, METHODS, MODEL_NAME
from smollm_lora.four_methods import run_four_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/four_methods",
        help="Directory for metrics, curves, generations, adapters, and figures.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
        help="Subset of recipes to run (default: all four).",
    )
    parser.add_argument("--epochs", type=int, default=FOUR_METHODS["epochs"])
    parser.add_argument("--learning-rate", type=float, default=FOUR_METHODS["learning_rate"])
    parser.add_argument("--max-length", type=int, default=FOUR_METHODS["max_length"])
    parser.add_argument("--max-new-tokens", type=int, default=FOUR_METHODS["max_new_tokens"])
    parser.add_argument("--seed", type=int, default=FOUR_METHODS["seed"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip writing the six comparison plots.",
    )
    args = parser.parse_args()

    train_settings = {
        **FOUR_METHODS,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    metrics = run_four_methods(
        Path(args.output),
        method_keys=args.methods,
        methods=METHODS,
        train_settings=train_settings,
        model_name=args.model,
        make_figures=not args.no_figures,
    )
    print(metrics[[
        "method",
        "trainable_params",
        "frozen_base_memory_mb",
        "test_loss",
        "train_seconds",
    ]])


if __name__ == "__main__":
    main()
