import os
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader, Dataset, Subset
import random


class DataLoaderFactory:
    def __init__(self, image_size=224, batch_size=16, num_workers=4, pin_memory=True):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def _transform_train(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0, 15)),
            transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1)),
            transforms.ToTensor(),
        ])

    def _transform_test(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def get_loaders(self, input_dir, val_split=0.2, shuffle=True):
        """
        Split the dataset into training and validation sets *before* applying transformations.

        Args:
            input_dir (str): Directory containing the image dataset.
            val_split (float): Fraction of the dataset to use for validation.
            shuffle (bool): Whether to shuffle the dataset before splitting.

        Returns:
            tuple: train_loader, val_loader
        """
        # Gather all file paths
        dataset = datasets.ImageFolder(root=input_dir)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        if shuffle:
            random.seed(42)  # Ensure reproducibility
            random.shuffle(indices)

        # Split indices
        val_size = int(val_split * dataset_size)
        train_indices, val_indices = indices[val_size:], indices[:val_size]

        # Subset datasets without transformations
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # Apply transformations to subsets
        train_dataset.dataset = datasets.ImageFolder(root=input_dir, transform=self._transform_train())
        val_dataset.dataset = datasets.ImageFolder(root=input_dir, transform=self._transform_test())

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=self.pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=self.pin_memory)

        return train_loader, val_loader
