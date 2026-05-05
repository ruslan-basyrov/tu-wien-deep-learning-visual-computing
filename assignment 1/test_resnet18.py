# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2

from pathlib import Path
from torchvision.models import resnet18
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.wandb_logger import WandBLogger
from config import DATA_DIR, MODEL_SAVE_DIR
from torch.utils.data import DataLoader


def test(args):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_data = CIFAR10Dataset(DATA_DIR, Subset.TEST, transform=transform)
    test_data_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = torch.nn.CrossEntropyLoss()

    model_paths = sorted({
        *Path(MODEL_SAVE_DIR).glob("resnet18*/model_best.pth"),
        *Path(MODEL_SAVE_DIR).glob("resnet18/*/model_best.pth"),
    })

    if not model_paths:
        print("No resnet18 models found in saved_models/")
        return

    for model_path in model_paths:
        print(f"\n=== {model_path.parent.name} ===")

        model = DeepClassifier(resnet18())
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

        logger = WandBLogger(enabled=True, run_name=model_path.parent.name, group="resnet18", resume="allow")
        logger.log({
            "test/loss": test_loss,
            "test/accuracy": accuracy,
            "test/per_class_accuracy": per_class_accuracy,
        })
        logger.finish()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Test ResNet18 on CIFAR-10")
    args.add_argument(
        "-d", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0

    test(args)
