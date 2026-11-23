"""Score a checkpoint with test loss, and optionally print the prompt suite."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs import LORA, MODEL_NAME
from smollm_lora.data import load_conversations, load_model, load_tokenizer
from smollm_lora.evaluate import generate_replies, validation_loss
from smollm_lora.lora import load_adapters, prepare_lora_model
from smollm_lora.prompts import PROMPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="Public model name, or a local directory from scripts/train_full.py.",
    )
    parser.add_argument(
        "--adapters",
        default=None,
        help="Path to lora_adapters.pt. The model argument should be the base checkpoint.",
    )
    parser.add_argument("--prompts", action="store_true", help="Also print the 10 prompt replies.")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model if args.adapters is None else MODEL_NAME)
    model = load_model(args.model if args.adapters is None else MODEL_NAME)
    if args.adapters is not None:
        prepare_lora_model(model, LORA["rank"], LORA["alpha"], LORA["target_modules"])
        load_adapters(model, args.adapters)
        tokenizer = load_tokenizer(str(Path(args.adapters).resolve().parent))

    dataset = load_conversations()
    loss = validation_loss(model, tokenizer, dataset)
    print("test loss:", float(loss))

    if args.prompts:
        for messages, reply in zip(PROMPTS, generate_replies(model, tokenizer, PROMPTS)):
            print("user:", messages[0]["content"])
            print(reply)
            print()


if __name__ == "__main__":
    main()
