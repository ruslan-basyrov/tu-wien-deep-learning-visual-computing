import pickle
from typing import Tuple
import numpy as np
from pathlib import Path

from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        Labels should be returned either as a Python list of ints or as a
        numpy array with dtype int64.
        """
        
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict

        if not Path(self.fdir).is_dir():
            raise ValueError(f"{self.fdir} is not a directory")

        subset_files = {
            Subset.TRAINING:   ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"],
            Subset.VALIDATION: ["data_batch_5"],
            Subset.TEST:       ["test_batch"],
        }
        files = subset_files[self.subset]
        
        all_images = []
        all_labels = []
        for file in files:
            fpath = self.fdir / file
            if not fpath.exists():
                raise ValueError(f"Missing file: {fpath}")
            d = unpickle(fpath)
            N = len(d[b"labels"])
            all_images.append(d[b"data"].reshape(N, 3, 32, 32).transpose(0, 2, 3, 1))
            all_labels.extend(d[b"labels"])

        images = np.concatenate(all_images, axis=0).astype(np.uint8)
        labels = np.array(all_labels, dtype=np.int64)
        return images, labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds")
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(self.classes)
