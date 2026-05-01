import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset


class TestCIFAR10Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_ds = CIFAR10Dataset(DATA_DIR, Subset.TRAINING)
        cls.val_ds = CIFAR10Dataset(DATA_DIR, Subset.VALIDATION)
        cls.test_ds = CIFAR10Dataset(DATA_DIR, Subset.TEST)

    def test_lengths(self):
        self.assertEqual(len(self.train_ds), 40000)
        self.assertEqual(len(self.val_ds), 10000)
        self.assertEqual(len(self.test_ds), 10000)

    def test_num_classes(self):
        self.assertEqual(self.train_ds.num_classes(), 10)

    def test_image_shape(self):
        img, _ = self.train_ds[0]
        self.assertEqual(img.shape, (32, 32, 3))

    def test_image_dtype(self):
        img, _ = self.train_ds[0]
        self.assertEqual(img.dtype, np.uint8)

    def test_first_ten_labels(self):
        labels = [self.train_ds[i][1] for i in range(10)]
        self.assertEqual(labels, [6, 9, 9, 4, 1, 1, 2, 7, 8, 3])

    def test_index_error(self):
        with self.assertRaises(IndexError):
            self.train_ds[99999]

    def test_value_error_bad_dir(self):
        with self.assertRaises(ValueError):
            CIFAR10Dataset(Path("/nonexistent/path"), Subset.TRAINING)


if __name__ == "__main__":
    unittest.main()
