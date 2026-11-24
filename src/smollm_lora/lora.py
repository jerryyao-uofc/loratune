"""A small LoRA layer, plus the helpers that attach and freeze it."""

import os
from typing import Iterable, List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Original linear layer plus a rank-r update.

    The update is ``scaling * B @ A``. Classic LoRA uses ``scaling = alpha / r``.
    Rank-stabilized LoRA (rsLoRA) uses ``scaling = alpha / sqrt(r)``.
    ``A`` starts random. ``B`` starts at zero, so the update starts at zero.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        in_features: int,
        out_features: int,
        r: int,
        alpha: int,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.use_rslora = use_rslora
        if use_rslora:
            self.scaling = self.alpha / (self.r ** 0.5)
        else:
            self.scaling = self.alpha / self.r

        weight_dtype = base_layer.weight.dtype
        weight_device = base_layer.weight.device
        self.A = nn.Parameter(
            torch.randn(r, in_features, dtype=weight_dtype, device=weight_device)
        )
        self.B = nn.Parameter(
            torch.zeros(out_features, r, dtype=weight_dtype, device=weight_device)
        )

    def forward(self, x):
        base_output = self.base_layer(x)
        # x @ A.T compresses each token to r numbers.
        # That result @ B.T expands those r numbers back to out_features.
        lora_update = (x @ self.A.T) @ self.B.T
        return base_output + self.scaling * lora_update


def replace_target_with_lora_layer(
    model,
    r: int,
    alpha: int,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    use_rslora: bool = False,
) -> None:
    """Replace each named linear layer with a LoRALinear that wraps it."""
    targets = set(target_modules)
    for name, module in model.named_modules():
        layer_name = name.split(".")[-1]
        if layer_name not in targets:
            continue
        parent_module = model
        for part in name.split(".")[:-1]:
            parent_module = getattr(parent_module, part)
        lora_layer = LoRALinear(
            module,
            module.in_features,
            module.out_features,
            r,
            alpha,
            use_rslora=use_rslora,
        )
        setattr(parent_module, layer_name, lora_layer)


def set_requires_grad_for_lora(model) -> None:
    """Train A and B. Leave every other parameter frozen."""
    for name, parameter in model.named_parameters():
        parameter_name = name.split(".")[-1]
        parameter.requires_grad = parameter_name in ("A", "B")


def count_trainable_parameters(model) -> int:
    total = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            total += parameter.numel()
    return total


def save_adapters(model, output_dir: str) -> str:
    """Save only the A and B matrices."""
    os.makedirs(output_dir, exist_ok=True)
    adapters = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.endswith(".A") or name.endswith(".B")
    }
    path = os.path.join(output_dir, "lora_adapters.pt")
    torch.save(adapters, path)
    return path


def load_adapters(model, adapter_path: str):
    state = torch.load(adapter_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


def prepare_lora_model(
    model,
    rank: int,
    alpha: int,
    target_modules: List[str],
    use_rslora: bool = False,
):
    replace_target_with_lora_layer(
        model, rank, alpha, target_modules, use_rslora=use_rslora
    )
    set_requires_grad_for_lora(model)
    return model
