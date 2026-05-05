import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from tqdm import tqdm

from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.metrics import Accuracy
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.models.cnn import YourCNN
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
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(not args.cpu and torch.cuda.is_available()),
    )

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if not Path(args.path_to_trained_model).exists():
        raise FileNotFoundError(f"Trained model not found: {args.path_to_trained_model}")

    model = DeepClassifier(YourCNN())
    model.load(args.path_to_trained_model)
    model.to(device)
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    test_metric = Accuracy(classes=test_data.classes)

    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            test_metric.update(outputs, labels)

    print(f"test loss: {total_loss / len(test_loader):.6f}")
    print(test_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Your CNN on CIFAR-10")
    parser.add_argument("-d", "--gpu_id", default="0", type=str, help="index of which GPU to use")
    parser.add_argument("--cpu", action="store_true", help="force CPU even if GPU is available")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int, help="Number of DataLoader workers")
    parser.add_argument(
        "--path_to_trained_model",
        default=str(Path(MODEL_SAVE_DIR) / "yourCNN" / "baseline" / "model_best.pth"),
        type=str,
    )

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    test(args)
