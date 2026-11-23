"""Test loss on held-out chats, and generation on the prompt suite."""

from typing import List

import torch
from torch.utils.data import DataLoader

from smollm_lora.template import apply_chat_template
from smollm_lora.train import pick_device


def validation_loss(model, tokenizer, dataset):
    """Mean next-token loss on the test split.

    Each conversation is scored alone, so padding never enters the average.
    ``labels`` is a copy of the token ids: the loss is the usual causal
    language-model loss over the whole conversation.
    """
    device = pick_device()

    def tokenize_row(row):
        return tokenizer(apply_chat_template(row["messages"]))

    tokenized = dataset.map(tokenize_row)
    tokenized = tokenized.map(lambda row: {"labels": row["input_ids"]})
    tokenized = tokenized.select_columns(["input_ids", "labels", "attention_mask"])
    tokenized.set_format("torch")

    loader = DataLoader(tokenized["test"], batch_size=1)
    model.to(device)
    losses = []
    with torch.no_grad():
        for batch in loader:
            output = model(**{key: value.to(device) for key, value in batch.items()})
            losses.append(output.loss)
    return sum(losses) / len(losses)


def assistant_prompt(messages: List) -> str:
    """Prompt string that stops where the assistant should start speaking."""
    return apply_chat_template(messages) + "\n<|im_start|><|assistant|>: "


def generate_replies(model, tokenizer, prompts: List, max_new_tokens: int = 50):
    """Generate up to ``max_new_tokens`` new tokens for each prompt."""
    device = pick_device()
    model.to(device)
    model.eval()
    replies = []
    for messages in prompts:
        prompt = assistant_prompt(messages)
        tokenized = tokenizer(prompt, return_tensors="pt")
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            output = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
            )
        replies.append(tokenizer.decode(output[0]))
    return replies
