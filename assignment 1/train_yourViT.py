import argparse
import hashlib
import json
import os
from itertools import product
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2

from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.models.vit import ViT
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.wandb_logger import WandBLogger
from config import DATA_DIR, MODEL_SAVE_DIR


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def make_run_name(params: dict) -> str:
    readable = f"yourvit_{params['augmentation']}_wd{params['weight_decay']}_dropout{params['dropout']}"
    hash_str = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{readable}_{hash_str}"


def train(args):
    transforms = {
        "none": v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        ),
        "basic": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        ),
        "strong": v2.Compose(
            [
                v2.ToImage(),
                v2.RandomCrop(32, padding=4),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                v2.RandomErasing(p=0.25),
            ]
        ),
    }

    grid = {
        "augmentation": ["none", "basic", "strong"],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "dropout":      [0.05, 0.1, 0.2],
    }

    val_data = CIFAR10Dataset(DATA_DIR, Subset.VALIDATION, transform=transforms["none"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model_save_dir = Path(MODEL_SAVE_DIR) / "yourViT"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    for augmentation, weight_decay, dropout in product(*grid.values()):
        run_name = make_run_name({
            "augmentation": augmentation,
            "weight_decay": weight_decay,
            "dropout": dropout,
        })

        train_data = CIFAR10Dataset(DATA_DIR, Subset.TRAINING, transform=transforms[augmentation])
        model = DeepClassifier(ViT(dropout=dropout))
        model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2e-3, weight_decay=weight_decay, amsgrad=True
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs
        )

        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)

        exp_save_dir = model_save_dir / run_name
        exp_save_dir.mkdir(exist_ok=True)

        logger = WandBLogger(
            enabled=True,
            run_name=run_name,
            config={
                "augmentation": augmentation,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
            },
            group="yourViT",
        )

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
            exp_save_dir,
            batch_size=args.batch_size,
            val_frequency=5,
            num_workers=8,
            prefetch_factor=4,
            logger=logger,
        )

        trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train Your ViT on CIFAR-10")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 60
    args.batch_size = 512

    train(args)
