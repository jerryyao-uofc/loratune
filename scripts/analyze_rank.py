"""Plot how much of the full fine-tune update a low-rank matrix can rebuild."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from configs import MODEL_NAME
from smollm_lora.analysis import delta_reconstruction_frame, plot_reconstruction_error
from smollm_lora.data import load_model

RANKS = [1, 8, 32, 64, 128, 256, 576]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", required=True, help="Local directory from scripts/train_full.py.")
    parser.add_argument("--output", default="outputs/rank_reconstruction.png")
    args = parser.parse_args()

    base = load_model(MODEL_NAME)
    finetuned = load_model(args.finetuned)
    frame = delta_reconstruction_frame(base, finetuned, RANKS)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_reconstruction_error(frame, save_path=args.output)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
