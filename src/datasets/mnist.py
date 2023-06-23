import torch.utils.data as data
from torchvision import datasets

MNIST_DIR = "data/mnist/"

class MNIST(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 1

    def __init__(
            self, 
            train=True, 
            image_transforms=None, 
        ):
        super().__init__()
        self.dataset = datasets.mnist.MNIST(
            MNIST_DIR, 
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