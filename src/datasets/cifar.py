import torch.utils.data as data
from torchvision import datasets

CIFAR10_DIR = "data/cifar10/"
CIFAR100_DIR = "data/cifar100/"

class CIFAR10(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        self.dataset = datasets.CIFAR10(
            CIFAR10_DIR, 
            train=train,
            download=True,
            transform=image_transforms,
        )

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        data = [index, img_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)


class CIFAR100(data.Dataset):
    NUM_CLASSES = 100
    NUM_CHANNELS = 3

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        self.dataset = datasets.CIFAR100(
            CIFAR100_DIR, 
            train=train,
            download=True,
            transform=image_transforms,
        )

    def __getitem__(self, index):
        img_data, label = self.dataset.__getitem__(index)
        data = [index, img_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.dataset)