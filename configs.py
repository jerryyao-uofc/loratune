"""Settings for the run that produced the recorded LoRATune results."""

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
