"""
Central configuration for machine-dependent paths used in assignment scripts.

Students should adapt these paths to their local setup.
"""

from pathlib import Path

# Path to extracted CIFAR-10 python files (directory containing data_batch_1 ... test_batch)
DATA_DIR = Path("/path/to/cifar-10-batches-py")

# Optional logging directory (for wandb/tensorboard/custom logs)
LOG_DIR = Path("logs")

# Directory where trained model checkpoints are stored
MODEL_SAVE_DIR = Path("saved_models")
