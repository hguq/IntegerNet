"""
Load cifar10 dataset.
"""
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from .util import *


def get_cifar10(dataset_path, batch_size):
    """
    return cifar10 train & test dataloader
    :param dataset_path: dataset path
    :param batch_size: batch size
    :return: cifar10 train & test dataloader
    """
    train_set = CIFAR10(dataset_path + "cifar10", train=True, transform=cifar_train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)

    test_set = CIFAR10(dataset_path + "cifar10", train=False, transform=cifar_test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader
