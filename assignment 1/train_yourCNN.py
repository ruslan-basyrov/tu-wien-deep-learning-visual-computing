import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2

from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.models.cnn import YourCNN
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.wandb_logger import WandBLogger
from config import DATA_DIR, MODEL_SAVE_DIR


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

CNN_EXPERIMENTS = [
    {
        "experiment_name": "cnn_basic_wd1e-4_dropout02",
        "augmentation": "basic",
        "weight_decay": 1e-4,
        "dropout": 0.2,
    },
    {
        "experiment_name": "cnn_strong_wd1e-4_dropout03",
        "augmentation": "strong",
        "weight_decay": 1e-4,
        "dropout": 0.3,
    },
    {
        "experiment_name": "cnn_basic_wd1e-3_dropout03",
        "augmentation": "basic",
        "weight_decay": 1e-3,
        "dropout": 0.3,
    },
    {
        "experiment_name": "cnn_none_wd0_dropout01",
        "augmentation": "none",
        "weight_decay": 0.0,
        "dropout": 0.1,
    },
]


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

    train_transform = transforms[args.augmentation]
    val_transform = transforms["none"]

    train_data = CIFAR10Dataset(DATA_DIR, Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(DATA_DIR, Subset.VALIDATION, transform=val_transform)

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = DeepClassifier(YourCNN(dropout_p=args.dropout))
    model.to(device)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum
        )

    if args.scheduler == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    elif args.scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)

    loss_fn = torch.nn.CrossEntropyLoss()
    train_metric = Accuracy(classes=train_data.classes)
    val_metric = Accuracy(classes=val_data.classes)

    model_save_dir = Path(MODEL_SAVE_DIR) / "yourCNN" / args.experiment_name
    model_save_dir.mkdir(parents=True, exist_ok=True)

    logger = None
    if args.wandb:
        logger = WandBLogger(
            enabled=True,
            run_name=args.experiment_name,
            config={
                "optimizer": args.optimizer,
                "scheduler": args.scheduler,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "augmentation": args.augmentation,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
            },
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
        model_save_dir,
        batch_size=args.batch_size,
        val_frequency=args.val_frequency,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.disable_persistent_workers,
        logger=logger,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Your CNN on CIFAR-10")
    parser.add_argument("-d", "--gpu_id", default="0", type=str, help="index of which GPU to use")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if GPU is available")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--amsgrad", action="store_true", help="Use amsgrad with AdamW")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="Optimizer to use",
    )
    parser.add_argument(
        "--scheduler",
        choices=["exponential", "cosine", "step", "none"],
        default="exponential",
        help="Learning rate scheduler",
    )
    parser.add_argument("--gamma", default=0.9, type=float, help="Scheduler decay factor")
    parser.add_argument("--t_max", default=30, type=int, help="T_max for cosine annealing")
    parser.add_argument("--step_size", default=10, type=int, help="Step size for StepLR")
    parser.add_argument(
        "--augmentation",
        choices=["none", "basic", "strong"],
        default="strong",
        help="Data augmentation policy",
    )
    parser.add_argument("--dropout", default=0.3, type=float, help="Dropout probability in classifier")
    parser.add_argument("--val_frequency", default=5, type=int)
    parser.add_argument("--num_workers", default=8, type=int, help="Number of DataLoader workers")
    parser.add_argument("--prefetch_factor", default=2, type=int, help="DataLoader prefetch factor")
    parser.add_argument("--disable_persistent_workers", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--experiment_name", default="baseline", type=str, help="Experiment name for saving and logging")
    parser.add_argument(
        "--run_experiments",
        action="store_true",
        help="Run the predefined CNN experiment set sequentially",
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.run_experiments:
        for experiment in CNN_EXPERIMENTS:
            for key, value in experiment.items():
                setattr(args, key, value)
            train(args)
    else:
        train(args)
