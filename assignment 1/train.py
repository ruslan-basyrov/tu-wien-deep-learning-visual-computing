# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from assignment_1_code.models.class_model import (
    DeepClassifier,
)  # etc. change to your model
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR


def train(args):

    # Implement this function so that it trains a specific model as described in the instruction.md file
    # feel free to change the code snippets given here, they are just to give you an initial structure
    # but do not have to be used if you want to do it differently
    # For device handling you can take a look at pytorch documentation

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use config.py for all machine-dependent paths, e.g. DATA_DIR.
    train_data = CIFAR10Dataset(DATA_DIR, ...)
    val_data = ...

    device = ...

    model = DeepClassifier(...)
    model.to(device)
    optimizer = ...
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)
    val_frequency = 5

    model_save_dir = Path(MODEL_SAVE_DIR)
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = ...

    trainer = ImgClassificationTrainer(
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        args.num_epochs,
        model_save_dir,
        batch_size=128,  # feel free to change
        val_frequency=val_frequency,
    )
    trainer.train()


if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 30

    train(args)
