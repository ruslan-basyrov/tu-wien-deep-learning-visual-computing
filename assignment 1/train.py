# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path

from assignment_1_code.models.class_model import (
    DeepClassifier,
)  # etc. change to your model
from assignment_1_code.metrics import Accuracy
from assignment_1_code.trainer import ImgClassificationTrainer
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR
from torchvision.models import resnet18
from itertools import product
from assignment_1_code.wandb_logger import WandBLogger
import hashlib, json

def make_run_name(params: dict) -> str:
    readable = f"resnet18_{params['optimizer_name']}_batch_size_of_{params['batch_size']}_num_epochs_of_{params['num_epochs']}_{params['augmentation']}_weight_decay_of_{params['weight_decay']}_{params['scheduler']}_scheduler"
    hash_str = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:6]
    return f"{readable}_{hash_str}"


def train(args):

    # Implement this function so that it trains a specific model as described in the instruction.md file
    # feel free to change the code snippets given here, they are just to give you an initial structure
    # but do not have to be used if you want to do it differently
    # For device handling you can take a look at pytorch documentation

    # train_transform = v2.Compose(
    #     [
    #         v2.ToImage(),
    #         v2.ToDtype(torch.float32, scale=True),
    #         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    train_transforms = {
        "no_augmentation": v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        "augmentation": v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            v2.RandomCrop(32, padding=4),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(),
        ]),
    }

    grid = {
        "augmentation":   ["augmentation", "no_augmentation"],
        "weight_decay":   [0.0, 1e-3],
        "scheduler":      ["exponential", "cosine"],
        "optimizer_name": ["adamw", "sgd"],
        "batch_size":     [512, 1024],
        "num_epochs":     [20, 30],
    }

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use config.py for all machine-dependent paths, e.g. DATA_DIR.
    # train_data = CIFAR10Dataset(DATA_DIR, Subset.TRAINING, transform=train_transform)
    val_data = CIFAR10Dataset(DATA_DIR, Subset.VALIDATION, transform=val_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_save_dir = Path(MODEL_SAVE_DIR)
    model_save_dir.mkdir(exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss()


    for augmentation, weight_decay, scheduler, optimizer_name, \
            batch_size, num_epochs in product(*grid.values()):
        transform = train_transforms[(augmentation)]
        run_name = make_run_name({
            "augmentation": augmentation, "weight_decay": weight_decay, 
            "scheduler": scheduler, "optimizer_name": optimizer_name,
            "batch_size": batch_size, "num_epochs": num_epochs
        })

        train_data = CIFAR10Dataset(DATA_DIR, Subset.TRAINING, transform=transform)
        # model = DeepClassifier(resnet18())
        model = DeepClassifier(resnet18())
        model.to(device)

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay, amsgrad=True)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay, momentum=0.9)
        if scheduler == "exponential":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)

        exp_save_dir = model_save_dir / run_name
        exp_save_dir.mkdir(exist_ok=True)

        
        wandb_config = {
            "augmentation": augmentation, "weight_decay": weight_decay,
            "scheduler": scheduler, "optimizer": optimizer_name,
            "batch_size": batch_size, "num_epochs": num_epochs
        }
        logger = WandBLogger(enabled=True, run_name=run_name, config=wandb_config)
        trainer = ImgClassificationTrainer(
            model, optimizer, loss_fn, lr_scheduler,
            train_metric, val_metric,
            train_data, val_data,
            device, num_epochs,
            exp_save_dir,
            batch_size=batch_size,
            val_frequency=5,
            logger=logger
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
