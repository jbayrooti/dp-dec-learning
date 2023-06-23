from torchvision import transforms
import random
from PIL import ImageFilter
from src.datasets.mnist import MNIST
from src.datasets.cifar import CIFAR10, CIFAR100

DATASET = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
}

def get_image_datasets(dataset_name):
    train_transforms, test_transforms = load_image_transforms(dataset_name)
    train_dataset = DATASET[dataset_name](
        train=True,
        image_transforms=train_transforms
    )
    val_dataset = DATASET[dataset_name](
        train=False,
        image_transforms=test_transforms,
    )
    return train_dataset, val_dataset

def load_image_transforms(dataset_name, resize_to_32=True):
    mean, std = get_data_mean_and_stdev(dataset_name)
    if resize_to_32:
        train_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return train_transforms, test_transforms

def get_data_mean_and_stdev(dataset):
    if dataset == 'mnist':
        mean = [0.1307,]
        std  = [0.3081,]
    elif dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif dataset == 'cifar100':
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
    else:
        raise Exception(f'Dataset {dataset} not supported.')
    return mean, std

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
