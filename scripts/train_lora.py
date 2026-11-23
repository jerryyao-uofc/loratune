"""LoRA fine-tune of SmolLM2 on SmolTalk everyday conversations."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs import LORA, MODEL_NAME
from smollm_lora.data import load_conversations, load_model, load_tokenizer
from smollm_lora.lora import count_trainable_parameters
from smollm_lora.train import train_lora


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/lora", help="Directory for the adapter file.")
    parser.add_argument("--rank", type=int, default=LORA["rank"])
    parser.add_argument("--alpha", type=int, default=LORA["alpha"])
    parser.add_argument("--learning-rate", type=float, default=LORA["learning_rate"])
    parser.add_argument("--epochs", type=int, default=LORA["epochs"])
    parser.add_argument("--max-length", type=int, default=LORA["max_length"])
    args = parser.parse_args()

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME)
    dataset = load_conversations()
    print("before LoRA:", count_trainable_parameters(model))
    train_lora(
        model,
        tokenizer,
        dataset,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=LORA["target_modules"],
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_length=args.max_length,
        output_dir=args.output,
    )
    print("after LoRA:", count_trainable_parameters(model))


if __name__ == "__main__":
    main()
