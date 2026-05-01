import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """

        ## TODO implement
        pass

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        pass

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        ## TODO implement
        pass

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        ## TODO implement
        pass
