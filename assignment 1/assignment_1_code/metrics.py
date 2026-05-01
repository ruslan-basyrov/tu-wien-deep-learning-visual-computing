from abc import ABCMeta, abstractmethod
import torch
from typing import Dict


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0
        self.per_class_accuracies = {classname: 0 for classname in self.classes}

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """
        if len(prediction.shape) != 2 or len(target.shape) != 1:
            raise ValueError("prediction must be (B, C), target must be (B,)")\

        predicted = torch.argmax(prediction, dim=1)
        correct_predictions = predicted == target

        self.n_matching += correct_predictions.sum().item()
        self.n_total += len(correct_predictions)

        for i, classname in enumerate(self.classes):
            class_mask = target == i
            self.total_pred[classname] += class_mask.sum().item()
            self.correct_pred[classname] += (correct_predictions & class_mask).sum().item()


    def __str__(self):
        """
        Return a string representation of the performance including:
        - overall accuracy
        - mean per-class accuracy
        - individual per-class accuracies for all classes
        """
        accuracy = self.accuracy()
        mean_accuracy = self.per_class_accuracies()

        lines = [f"accuracy: {accuracy:.4f}", f"per class accuracy: {mean_accuracy:.4f}"]
        for c, a in self.per_class_accuracy():
            lines.append(f"Accuracy for class: {c} is {a:.4f}")

        return "\n".join(lines)

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        if self.n_total == 0:
            return 0.0
        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the mean per-class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        Saves the individual per-class accuracies in self.per_class_accuracies as a dict mapping class name to accuracy.
        """
        if self.n_total == 0:
            return 0.0
        
        self.per_class_accuracies = {
            c: (self.correct_pred[c] / self.total_pred[c] if self.total_pred[c] > 0 else 0.0)
            for c in self.classes
        }

        return sum(self.per_class_accuracies.values()) / len(self.classes)
