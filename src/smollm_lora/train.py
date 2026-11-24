"""Training loops for a full weight update and for LoRA."""

import os

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from smollm_lora.data import make_train_loader, tokenize_train
from smollm_lora.lora import prepare_lora_model, save_adapters


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_epochs(model, loader, optimizer, device: str, epochs: int) -> None:
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}"):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("epoch", epoch + 1, "average loss", running_loss / len(loader))
    model.eval()


def train_full(model, tokenizer, dataset, learning_rate, batch_size, epochs, max_length, output_dir):
    """Update every weight. Matches the recorded full-model run."""
    device = pick_device()
    model.to(device)
    train_data = tokenize_train(dataset, tokenizer, max_length)
    loader = make_train_loader(train_data, batch_size, tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    _run_epochs(model, loader, optimizer, device, epochs)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def train_lora(
    model,
    tokenizer,
    dataset,
    rank,
    alpha,
    target_modules,
    learning_rate,
    epochs,
    max_length,
    output_dir,
    use_rslora: bool = False,
):
    """Freeze the base model and train the rank-r adapters."""
    device = pick_device()
    prepare_lora_model(model, rank, alpha, target_modules, use_rslora=use_rslora)
    model.to(device)
    train_data = tokenize_train(dataset, tokenizer, max_length)
    loader = make_train_loader(train_data, 1, tokenizer.pad_token_id)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=learning_rate)
    _run_epochs(model, loader, optimizer, device, epochs)

    os.makedirs(output_dir, exist_ok=True)
    save_adapters(model, output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir
