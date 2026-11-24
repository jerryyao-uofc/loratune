"""Settings for SmolLM2 LoRA runs and the four-method PEFT comparison."""

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "everyday-conversations"

FULL = {
    "learning_rate": 5e-5,
    "batch_size": 4,
    "epochs": 3,
    "max_length": 512,
}

LORA = {
    "rank": 8,
    "alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 1e-4,
    "batch_size": 1,
    "epochs": 3,
    "max_length": 512,
}

# Shared knobs for the four PEFT recipes (classic / QLoRA / all-linear / rsLoRA).
FOUR_METHODS = {
    "learning_rate": 1e-4,
    "batch_size": 1,
    "epochs": 3,
    "max_length": 512,
    "max_new_tokens": 50,
    "seed": 42,
}

ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Four production-style recipes compared in LoRA Lens.
# Classic and QLoRA share the attention adapter; all-linear and rsLoRA share
# the wider target list. Only one conceptual knob changes inside each pair.
METHODS = {
    "original_lora": {
        "label": "Classic LoRA",
        "r": 8,
        "alpha": 16,
        "target_modules": ATTENTION_TARGETS,
        "use_rslora": False,
        "load_in_4bit": False,
        "what_changed": "attention only, r=8, scale alpha/r, float base",
    },
    "qlora": {
        "label": "QLoRA",
        "r": 8,
        "alpha": 16,
        "target_modules": ATTENTION_TARGETS,
        "use_rslora": False,
        "load_in_4bit": True,
        "what_changed": "same adapter as classic, 4-bit NF4 frozen base",
    },
    "all_linear": {
        "label": "All-linear LoRA",
        "r": 16,
        "alpha": 16,
        "target_modules": ALL_LINEAR_TARGETS,
        "use_rslora": False,
        "load_in_4bit": False,
        "what_changed": "all linear layers, r=16, scale alpha/r",
    },
    "rslora": {
        "label": "rsLoRA",
        "r": 16,
        "alpha": 16,
        "target_modules": ALL_LINEAR_TARGETS,
        "use_rslora": True,
        "load_in_4bit": False,
        "what_changed": "same layers/rank as all_linear, scale alpha/sqrt(r)",
    },
}
