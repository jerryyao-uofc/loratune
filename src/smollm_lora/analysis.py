"""How much of a full fine-tune update a rank-r matrix can rebuild."""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch


def reconstruction_error(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """Frobenius error of the best rank-``rank`` approximation of ``weight``.

    The return value is a scalar tensor. If ``rank`` is larger than the
    matrix rank, the extra directions are dropped.
    """
    weight = weight.float()
    left_vectors, sizes, right_vectors = torch.linalg.svd(weight, full_matrices=False)

    matrix_rank = int(torch.linalg.matrix_rank(weight).item())
    kept = min(rank, matrix_rank)

    approximation = torch.zeros_like(weight)
    for index in range(kept):
        one_direction = torch.outer(left_vectors[:, index], right_vectors[index, :])
        approximation = approximation + sizes[index] * one_direction

    return torch.linalg.norm(weight - approximation)


def delta_reconstruction_frame(base_model, finetuned_model, ranks) -> pd.DataFrame:
    """Reconstruction error of W_base - W_finetuned for attention and MLP weights."""
    records = {}
    base_state = base_model.state_dict()
    tuned_state = finetuned_model.state_dict()
    for name in base_state:
        if "self_attn" not in name and "mlp" not in name:
            continue
        if base_state[name].ndim != 2:
            continue
        records[name] = {}
        delta = base_state[name] - tuned_state[name]
        for rank in ranks:
            records[name][rank] = float(reconstruction_error(delta, rank))
    frame = pd.DataFrame(records).reset_index().rename(columns={"index": "rank"})
    return frame


def plot_reconstruction_error(frame: pd.DataFrame, save_path: Optional[str] = None):
    """Six panels, one per projection. Darker lines are earlier layers."""
    figure, axes = plt.subplots(ncols=3, nrows=2, figsize=(10, 6))
    colormap = plt.colormaps["viridis"]
    num_lines = 30
    modules = ["k_proj", "q_proj", "o_proj", "v_proj", "up_proj", "gate_proj"]
    for index, module_name in enumerate(modules):
        axis = axes[index // 3][index % 3]
        columns = [column for column in frame.columns if module_name in column]
        for line_index, column in enumerate(columns):
            color = colormap(line_index / (num_lines - 1))
            axis.plot(frame["rank"], frame[column], color=color, marker=".")
        axis.set_title(module_name)
        axis.set_xlabel("rank")
        axis.set_ylabel("error")
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, dpi=150)
    return figure
