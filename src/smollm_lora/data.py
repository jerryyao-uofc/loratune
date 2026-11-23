"""Load SmolTalk and turn conversations into model batches."""

from typing import Callable, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from smollm_lora.template import apply_chat_template

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "everyday-conversations"


def load_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = "<|pad_token|>"
    return tokenizer


def load_model(model_name: str = MODEL_NAME):
    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)


def load_conversations():
    """Everyday-conversations split: 2,260 train rows and 119 test rows."""
    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def tokenize_messages(example: Dict, tokenizer, max_length: int) -> Dict:
    text = apply_chat_template(example["messages"])
    tokens = tokenizer(text, truncation=True, max_length=max_length)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def tokenize_train(dataset, tokenizer, max_length: int):
    train = dataset["train"].map(
        lambda example: tokenize_messages(example, tokenizer, max_length),
        remove_columns=dataset["train"].column_names,
    )
    train.set_format("torch")
    return train


def pad_batch(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad a batch of conversations. Label -100 marks positions the loss ignores."""
    longest = max(item["input_ids"].shape[0] for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad_n = longest - item["input_ids"].shape[0]
        input_ids.append(torch.cat([
            item["input_ids"].long(),
            torch.full((pad_n,), pad_id, dtype=torch.long),
        ]))
        attention_mask.append(torch.cat([
            item["attention_mask"].long(),
            torch.zeros(pad_n, dtype=torch.long),
        ]))
        labels.append(torch.cat([
            item["labels"].long(),
            torch.full((pad_n,), -100, dtype=torch.long),
        ]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def make_train_loader(train_data, batch_size: int, pad_id: int) -> DataLoader:
    if batch_size == 1:
        return DataLoader(train_data, batch_size=1, shuffle=True)
    collate: Callable = lambda batch: pad_batch(batch, pad_id)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)
