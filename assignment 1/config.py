"""
Central configuration for machine-dependent paths used in assignment scripts.

Students should adapt these paths to their local setup.
"""

from pathlib import Path

# Path to extracted CIFAR-10 python files (directory containing data_batch_1 ... test_batch)
DATA_DIR = Path("../cifar-10-batches-py")

# Optional logging directory (for wandb/tensorboard/custom logs)
LOG_DIR = Path("logs")

# Directory where trained model checkpoints are stored
MODEL_SAVE_DIR = Path("saved_models")

EXPERIMENTS = [
    {
        "name": "baseline",
        "augment": False,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "scheduler": "exponential",
    },
    {
        "name": "augment_only",
        "augment": True,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "scheduler": "exponential",
    },
    {
        "name": "augment_wd1e-4",
        "augment": True,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "scheduler": "exponential",
    },
    {
        "name": "augment_wd1e-2_cosine",
        "augment": True,
        "weight_decay": 1e-2,
        "optimizer": "adamw",
        "scheduler": "cosine",
    },
]
