import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2

from torch.utils.data import DataLoader
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.models.vit import ViT
from assignment_1_code.wandb_logger import WandBLogger
from config import DATA_DIR, MODEL_SAVE_DIR


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def test(args):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    test_data = CIFAR10Dataset(DATA_DIR, Subset.TEST, transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = torch.nn.CrossEntropyLoss()

    model_paths = sorted({
        *Path(MODEL_SAVE_DIR).glob("yourViT*/model_best.pth"),
        *Path(MODEL_SAVE_DIR).glob("yourViT/*/model_best.pth"),
    })

    if not model_paths:
        print("No yourViT models found in saved_models/")
        return

    for model_path in model_paths:
        print(f"\n=== {model_path.parent.name} ===")

        model = DeepClassifier(ViT())
        model.load(str(model_path))
        model.to(device)

        test_metric = Accuracy(classes=test_data.classes)
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in test_data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                test_metric.update(outputs, labels)

        test_loss = total_loss / len(test_data_loader)
        accuracy = test_metric.accuracy()
        per_class_accuracy = test_metric.per_class_accuracy()

        print(f"test loss: {test_loss}")
        print(test_metric)

        logger = WandBLogger(enabled=True, run_name=model_path.parent.name, group="yourViT", resume="allow")
        logger.log({
            "test/loss": test_loss,
            "test/accuracy": accuracy,
            "test/per_class_accuracy": per_class_accuracy,
        })
        logger.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test Your ViT on CIFAR-10")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    test(args)
