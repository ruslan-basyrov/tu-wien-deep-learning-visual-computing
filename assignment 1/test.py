# Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os

from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from config import DATA_DIR, MODEL_SAVE_DIR


def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use config.py for machine-dependent paths, e.g. DATA_DIR and MODEL_SAVE_DIR.
    test_data = CIFAR10Dataset(DATA_DIR, ...)
    test_data_loader = ...

    device = ...
    num_test_data = len(test_data)

    model = DeepClassifier(...)
    model.load(args.path_to_trained_model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    # Below implement testing loop and print final loss
    # and metrics to terminal after testing is finished
    # ...


if __name__ == "__main__":
    # Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-d", "--gpu_id", default="5", type=str, help="index of which GPU to use"
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.gpu_id = 0
    args.path_to_trained_model = str(MODEL_SAVE_DIR / "ResNet_model_best.pth")

    test(args)
