"""
Load mnist dataset.
"""
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from os.path import join

from .util import *

mnist_transform = Compose([
    ToTensor(),
    Float2Int()
])


def get_mnist(dataset_path, batch_size, num_workers):
    """
    return mnist train & test dataloader.
    :param dataset_path: dataset path
    :param batch_size: batch size
    :return: mnist train & test dataloader
    """
    if num_workers is None:
        num_workers = 4
    train_set = MNIST(join(dataset_path, "mnist"), train=True, transform=mnist_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_set = MNIST(join(dataset_path, "mnist"), train=False, transform=mnist_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
