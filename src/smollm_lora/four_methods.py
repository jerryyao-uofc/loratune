"""Train and evaluate the four PEFT LoRA recipes side by side.

Uses Hugging Face PEFT (+ bitsandbytes for QLoRA). Recipe definitions live in
``configs.METHODS``. Scripts and the Colab notebook both call into this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

from smollm_lora.data import (
    DATASET_CONFIG,
    DATASET_NAME,
    MODEL_NAME,
    load_conversations,
    load_tokenizer,
    make_train_loader,
    tokenize_train,
)
from smollm_lora.evaluate import assistant_prompt
from smollm_lora.lora import count_trainable_parameters
from smollm_lora.prompts import PROMPTS
from smollm_lora.template import apply_chat_template
from smollm_lora.train import pick_device

try:
    from configs import FOUR_METHODS, METHODS
except ImportError:  # pragma: no cover - package imported without repo root on path
    FOUR_METHODS = {
        "learning_rate": 1e-4,
        "batch_size": 1,
        "epochs": 3,
        "max_length": 512,
        "max_new_tokens": 50,
        "seed": 42,
    }
    METHODS = {}


METHOD_COLORS = {
    "original_lora": "#475569",
    "qlora": "#0F766E",
    "all_linear": "#9A3412",
    "rslora": "#1D4ED8",
}

FULL_FT_TEST_LOSS = 0.883


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model):
    return next(p for p in model.parameters() if p.device.type != "meta").device


def estimate_frozen_base_memory_mb(model, load_in_4bit: bool) -> float:
    """Rough weight storage only: 4-bit ≈ 0.5 byte/param, float32 ≈ 4 bytes/param."""
    total = sum(p.numel() for p in model.parameters())
    bytes_per = 0.5 if load_in_4bit else 4.0
    return total * bytes_per / (1024 ** 2)


def build_peft_model(method_cfg: Dict, model_name: str = MODEL_NAME, device: Optional[str] = None):
    """Load the base model and attach a PEFT LoRA adapter for one recipe."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The four-method runner needs the `peft` package. "
            "Install with: pip install peft bitsandbytes accelerate"
        ) from exc

    device = device or pick_device()
    if method_cfg["load_in_4bit"]:
        if device != "cuda":
            raise RuntimeError("QLoRA (4-bit) requires a CUDA GPU with bitsandbytes.")
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model.to(device)

    lora_config = LoraConfig(
        r=method_cfg["r"],
        lora_alpha=method_cfg["alpha"],
        target_modules=list(method_cfg["target_modules"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=bool(method_cfg["use_rslora"]),
    )
    return get_peft_model(model, lora_config)


@torch.no_grad()
def peft_validation_loss(model, tokenizer, dataset) -> float:
    """Mean next-token loss on the test split, one conversation at a time."""
    model.eval()
    device = model_device(model)
    losses = []
    for row in tqdm(dataset["test"], desc="test loss", leave=False):
        text = apply_chat_template(row["messages"])
        batch = tokenizer(text, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        batch = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(model(**batch).loss.item()))
    return float(np.mean(losses))


def first_assistant_span(decoded: str) -> str:
    marker = "<|im_start|><|assistant|>:"
    if marker not in decoded:
        return decoded.strip()
    span = decoded.split(marker, 1)[1]
    if "<|im_end|>" in span:
        span = span.split("<|im_end|>", 1)[0]
    return span.strip()


@torch.no_grad()
def peft_generate_replies(
    model,
    tokenizer,
    prompts: Sequence = PROMPTS,
    max_new_tokens: int = 50,
) -> List[Dict]:
    model.eval()
    device = model_device(model)
    rows = []
    for index, messages in enumerate(prompts):
        prompt = assistant_prompt(messages)
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=False)
        rows.append({
            "prompt_id": index,
            "prompt": messages[0]["content"],
            "full_decode": decoded,
            "first_assistant_span": first_assistant_span(decoded),
        })
    return rows


def train_one_method(
    method_key: str,
    output_dir: Path,
    *,
    methods: Optional[Dict] = None,
    train_settings: Optional[Dict] = None,
    tokenizer=None,
    dataset=None,
    train_loader: Optional[DataLoader] = None,
    model_name: str = MODEL_NAME,
) -> Dict:
    """Train one recipe, evaluate, and write adapter + generations under output_dir."""
    methods = methods or METHODS
    train_settings = train_settings or FOUR_METHODS
    cfg = methods[method_key]
    output_dir = Path(output_dir)
    adapters_root = output_dir / "adapters"
    generations_root = output_dir / "generations"
    adapters_root.mkdir(parents=True, exist_ok=True)
    generations_root.mkdir(parents=True, exist_ok=True)

    if tokenizer is None:
        tokenizer = load_tokenizer(model_name)
    if dataset is None:
        dataset = load_conversations()
    if train_loader is None:
        train_data = tokenize_train(dataset, tokenizer, train_settings["max_length"])
        train_loader = make_train_loader(
            train_data, train_settings["batch_size"], tokenizer.pad_token_id
        )

    print("=" * 72)
    print("Training", cfg["label"], f"({method_key})")
    print(cfg["what_changed"])
    print("=" * 72)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = build_peft_model(cfg, model_name=model_name)
    device = model_device(model)
    trainable = count_trainable_parameters(model)
    frozen_mb = estimate_frozen_base_memory_mb(model, cfg["load_in_4bit"])

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_settings["learning_rate"],
    )
    total_steps = train_settings["epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    history = []
    start = time.time()
    model.train()
    for epoch in range(train_settings["epochs"]):
        running = 0.0
        for batch in tqdm(train_loader, desc=f"{method_key} epoch {epoch + 1}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
        epoch_loss = running / len(train_loader)
        history.append({"method": method_key, "epoch": epoch + 1, "train_loss": epoch_loss})
        print(f"epoch {epoch + 1}: train loss {epoch_loss:.4f}")

    train_seconds = time.time() - start
    test_loss = peft_validation_loss(model, tokenizer, dataset)
    generations = peft_generate_replies(
        model,
        tokenizer,
        PROMPTS,
        max_new_tokens=train_settings["max_new_tokens"],
    )

    adapter_dir = adapters_root / method_key
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    gen_path = generations_root / f"{method_key}.json"
    gen_path.write_text(json.dumps(generations, indent=2))

    del model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "method": method_key,
        "label": cfg["label"],
        "what_changed": cfg["what_changed"],
        "rank": cfg["r"],
        "alpha": cfg["alpha"],
        "use_rslora": cfg["use_rslora"],
        "load_in_4bit": cfg["load_in_4bit"],
        "target_modules": ",".join(cfg["target_modules"]),
        "trainable_params": trainable,
        "frozen_base_memory_mb": round(frozen_mb, 1),
        "updates_per_epoch": len(train_loader),
        "train_seconds": round(train_seconds, 1),
        "test_loss": test_loss,
        "prompts_total": len(PROMPTS),
        "history": history,
        "generations_path": str(gen_path),
        "adapter_dir": str(adapter_dir),
    }


def collect_generations_table(output_dir: Path, method_keys: Iterable[str]) -> pd.DataFrame:
    """Flatten per-method generation JSON files into one CSV-ready table."""
    rows = []
    for method_key in method_keys:
        path = Path(output_dir) / "generations" / f"{method_key}.json"
        for row in json.loads(path.read_text()):
            rows.append({
                "method": method_key,
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "first_assistant_span": row["first_assistant_span"],
                "addressed": "",
            })
    return pd.DataFrame(rows)


def save_comparison_figures(
    metrics_df: pd.DataFrame,
    curves_df: pd.DataFrame,
    figures_dir: Path,
    methods: Optional[Dict] = None,
) -> None:
    """Write the six LoRA Lens comparison plots."""
    methods = methods or METHODS
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    labels = {key: cfg["label"] for key, cfg in methods.items()}

    # 1) Test loss bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metrics_df))
    bars = ax.bar(
        x,
        metrics_df["test_loss"],
        color=[METHOD_COLORS[m] for m in metrics_df["method"]],
        edgecolor="white",
        width=0.72,
    )
    ax.axhline(
        FULL_FT_TEST_LOSS,
        color="#B45309",
        linestyle="--",
        linewidth=1.4,
        label="Full fine-tune (recorded)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[m] for m in metrics_df["method"]], rotation=15, ha="right")
    ax.set_ylabel("Mean test loss")
    ax.set_title("Held-out next-token loss across four LoRA recipes")
    ax.legend(frameon=False)
    for bar, value in zip(bars, metrics_df["test_loss"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "01_test_loss.png", dpi=180)
    plt.close(fig)

    # 2) Trainable parameters vs test loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in metrics_df.iterrows():
        ax.scatter(
            row["trainable_params"] / 1e3,
            row["test_loss"],
            s=160,
            color=METHOD_COLORS[row["method"]],
            label=labels[row["method"]],
            zorder=3,
        )
        ax.annotate(
            labels[row["method"]],
            (row["trainable_params"] / 1e3, row["test_loss"]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )
    ax.set_xlabel("Trainable parameters (thousands)")
    ax.set_ylabel("Mean test loss")
    ax.set_title("Quality against adapter size")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "02_params_vs_loss.png", dpi=180)
    plt.close(fig)

    # 3) Training curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method_key, group in curves_df.groupby("method"):
        ax.plot(
            group["epoch"],
            group["train_loss"],
            marker="o",
            linewidth=2.2,
            color=METHOD_COLORS[method_key],
            label=labels[method_key],
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training loss")
    ax.set_title("Training loss by epoch")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "03_train_curves.png", dpi=180)
    plt.close(fig)

    # 4) Frozen base memory
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [labels[m] for m in metrics_df["method"]],
        metrics_df["frozen_base_memory_mb"],
        color=[METHOD_COLORS[m] for m in metrics_df["method"]],
        edgecolor="white",
    )
    ax.set_ylabel("Approx. frozen base weight memory (MB)")
    ax.set_title("Why QLoRA exists: storage of W, not the adapter")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(figures_dir / "04_frozen_memory.png", dpi=180)
    plt.close(fig)

    # 5) Pairwise test-loss gaps
    pairs = [
        ("original_lora", "qlora", "Quantization\n(classic vs QLoRA)"),
        ("original_lora", "all_linear", "Recipe\n(attention r=8 vs all-linear r=16)"),
        ("all_linear", "rslora", "Scale\n(alpha/r vs alpha/sqrt(r))"),
    ]
    loss_map = dict(zip(metrics_df["method"], metrics_df["test_loss"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pair_names = [pair[2] for pair in pairs]
    deltas = [loss_map[b] - loss_map[a] for a, b, _ in pairs]
    bar_colors = ["#0F766E" if d <= 0 else "#9A3412" for d in deltas]
    ax.bar(pair_names, deltas, color=bar_colors, edgecolor="white")
    ax.axhline(0, color="#334155", linewidth=1)
    ax.set_ylabel("Δ test loss (right minus left)")
    ax.set_title("Three controlled comparisons")
    for i, delta in enumerate(deltas):
        ax.text(
            i,
            delta + (0.005 if delta >= 0 else -0.01),
            f"{delta:+.3f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "05_pairwise_deltas.png", dpi=180)
    plt.close(fig)

    # 6) Wall-clock cost
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        [labels[m] for m in metrics_df["method"]],
        metrics_df["train_seconds"] / 60.0,
        color=[METHOD_COLORS[m] for m in metrics_df["method"]],
        edgecolor="white",
    )
    ax.set_xlabel("Training time (minutes)")
    ax.set_title("Wall-clock training cost on this GPU")
    fig.tight_layout()
    fig.savefig(figures_dir / "06_train_time.png", dpi=180)
    plt.close(fig)


def run_four_methods(
    output_dir: Path,
    method_keys: Optional[Sequence[str]] = None,
    *,
    methods: Optional[Dict] = None,
    train_settings: Optional[Dict] = None,
    model_name: str = MODEL_NAME,
    make_figures: bool = True,
) -> pd.DataFrame:
    """Train every requested recipe and write the comparison bundle."""
    methods = methods or METHODS
    if not methods:
        raise RuntimeError(
            "No METHODS dict available. Run via scripts/train_four_methods.py "
            "or put the repo root on sys.path so configs.METHODS imports."
        )
    train_settings = train_settings or FOUR_METHODS
    method_keys = list(method_keys or methods.keys())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapters").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "generations").mkdir(exist_ok=True)

    set_seed(train_settings.get("seed", 42))
    tokenizer = load_tokenizer(model_name)
    dataset = load_conversations()
    train_data = tokenize_train(dataset, tokenizer, train_settings["max_length"])
    train_loader = make_train_loader(
        train_data, train_settings["batch_size"], tokenizer.pad_token_id
    )

    results = []
    all_histories = []
    for method_key in method_keys:
        summary = train_one_method(
            method_key,
            output_dir,
            methods=methods,
            train_settings=train_settings,
            tokenizer=tokenizer,
            dataset=dataset,
            train_loader=train_loader,
            model_name=model_name,
        )
        all_histories.extend(summary.pop("history"))
        results.append(summary)
        pd.DataFrame(results).to_csv(output_dir / "metrics_partial.csv", index=False)
        pd.DataFrame(all_histories).to_csv(output_dir / "train_curves_partial.csv", index=False)
        print("Saved partial metrics for", method_key)

    metrics_df = pd.DataFrame(results)
    curves_df = pd.DataFrame(all_histories)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    curves_df.to_csv(output_dir / "train_curves.csv", index=False)

    generations_df = collect_generations_table(output_dir, method_keys)
    generations_df.to_csv(output_dir / "generations.csv", index=False)

    run_config = {
        "model": model_name,
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "learning_rate": train_settings["learning_rate"],
        "epochs": train_settings["epochs"],
        "batch_size": train_settings["batch_size"],
        "max_length": train_settings["max_length"],
        "max_new_tokens": train_settings["max_new_tokens"],
        "methods": {
            key: {
                **{k: v for k, v in cfg.items() if k != "target_modules"},
                "target_modules": list(cfg["target_modules"]),
            }
            for key, cfg in methods.items()
            if key in method_keys
        },
        "seed": train_settings.get("seed", 42),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    if make_figures:
        save_comparison_figures(metrics_df, curves_df, output_dir / "figures", methods=methods)

    return metrics_df
