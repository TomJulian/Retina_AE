from torchvision.transforms import v2 as transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision import datasets
from torch.utils.data import DataLoader

""" 
use example:
data_factory = DataLoaderFactory(image_size=224, batch_size=32, num_workers=8, pin_memory=True)
train_loader, test_loader = data_factory.get_loaders("/path/to/train", "/path/to/test")
"""


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

    def get_loaders(self, train_input_dir, test_input_dir):
        train_dataset = datasets.ImageFolder(root=train_input_dir, transform=self._transform_train())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                  num_workers=self.num_workers, pin_memory=self.pin_memory)

        test_dataset = datasets.ImageFolder(root=test_input_dir, transform=self._transform_test())
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                 num_workers=self.num_workers, pin_memory=self.pin_memory)

        return train_loader, test_loader
