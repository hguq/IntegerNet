"""
Load cifar100 dataset.
"""
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from os.path import join
from .util import *


def get_cifar100(dataset_path, batch_size):
    """
    return cifar100 train & test dataloader
    :param dataset_path: dataset path
    :param batch_size:  batch size
    :return: cifar100 train & test dataloder
    """
    train_set = CIFAR100(join(dataset_path, "cifar100"), train=True, transform=cifar_train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=8)

    test_set = CIFAR100(join(dataset_path, "cifar100"), train=False, transform=cifar_test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader
