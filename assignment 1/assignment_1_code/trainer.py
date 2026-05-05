import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
from torch.amp import autocast, GradScaler

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
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        logger = None,
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

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.val_frequency = val_frequency

        pin_memory = self.device.type == "cuda"
        loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["persistent_workers"] = persistent_workers

        self.train_loader = DataLoader(
            train_data,
            shuffle=True,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            val_data,
            shuffle=False,
            **loader_kwargs,
        )

        self.logger = logger or WandBLogger(enabled=False)
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        
        self.model.train()
        self.train_metric.reset()
        total_loss = 0.0

        for images, labels in tqdm(self.train_loader, desc=f"Train epoch {epoch_idx}"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            if self.use_amp:
                with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            self.train_metric.update(outputs.float(), labels)

        accuracy = self.train_metric.accuracy()
        mean_accuracy = self.train_metric.per_class_accuracy()
        mean_loss = total_loss / len(self.train_loader)
        print(f"______epoch {epoch_idx}\n{self.train_metric}")
        return accuracy, mean_accuracy, mean_loss


    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        
        self.model.eval()
        self.val_metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Val epoch {epoch_idx}"):
                images, labels = images.to(self.device), labels.to(self.device)
                if self.use_amp:
                    with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                self.val_metric.update(outputs.float(), labels)

        accuracy = self.val_metric.accuracy()
        mean_accuracy = self.val_metric.per_class_accuracy()
        mean_loss = total_loss / len(self.val_loader)
        print(f"______epoch {epoch_idx}\n{self.val_metric}")
        return accuracy, mean_accuracy, mean_loss




    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        t_start = time.time()
        best_mean_accuracy = 0.0

        for epoch in range(self.num_epochs):
            train_accuracy, train_mean_accuracy, train_mean_loss = self._train_epoch(epoch)
            self.logger.log({"train/accuracy": train_accuracy, 
                            "train/mean_accuracy": train_mean_accuracy, 
                            "train/mean_loss": train_mean_loss})
            if (epoch + 1) % self.val_frequency == 0:
                val_accuracy, val_mean_accuracy, val_mean_loss = self._val_epoch(epoch)
                self.logger.log({"val/accuracy": val_accuracy, 
                                "val/mean_accuracy": val_mean_accuracy, 
                                "val/mean_loss": val_mean_loss})
                    
                if val_mean_accuracy > best_mean_accuracy:
                    best_mean_accuracy = val_mean_accuracy
                    self.model.save(self.training_save_dir, suffix = "best")
            
            self.lr_scheduler.step()

        self.logger.log({"total_train_time_s": time.time() - t_start})
        self.logger.finish()
