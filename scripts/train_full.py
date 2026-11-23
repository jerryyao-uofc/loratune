"""Full-model fine-tune of SmolLM2 on SmolTalk everyday conversations."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs import FULL, MODEL_NAME
from smollm_lora.data import load_conversations, load_model, load_tokenizer
from smollm_lora.train import train_full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/full", help="Directory for the saved model.")
    parser.add_argument("--learning-rate", type=float, default=FULL["learning_rate"])
    parser.add_argument("--batch-size", type=int, default=FULL["batch_size"])
    parser.add_argument("--epochs", type=int, default=FULL["epochs"])
    parser.add_argument("--max-length", type=int, default=FULL["max_length"])
    args = parser.parse_args()

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    dataset = load_conversations()
    train_full(
        model,
        tokenizer,
        dataset,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
