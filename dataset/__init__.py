"""
A package to load all datasets.
"""
import matplotlib.pyplot as plt
from .cifar10 import *
from .cifar100 import *
from .mnist import *
from .imagenet import *

dataset_path = "/data/student/stu514-16/datasets/"
# dataset_path = "C:/projects/datasets/"


def get_loader(dataset, batch_size):
    """
    A function to get train dataloader & test dataloader
    :param dataset: String, like "Cifar10" or "Imagenet"
    :param batch_size: Int, batch size
    :return: train loader & test loader
    """
    return {"cifar10": get_cifar10,
            "cifar100": get_cifar100,
            "mnist": get_mnist,
            "imagenet": get_imagenet}[dataset.lower()](dataset_path, batch_size)


def test_datasets():
    """
    A function to test all datasets
    :return: None
    """
    train_loader, test_loader = get_loader("MNIST", 1)
    data, label = next(iter(train_loader))
    print("MNIST TRAIN:", data.max(), data.min())
    data, label = next(iter(test_loader))
    print("MNIST TEST:", data.max(), data.min())

    train_loader, test_loader = get_loader("CIFAR10", 1)
    data, label = next(iter(train_loader))
    print("CIFAR10 TRAIN:", data.max(), data.min())
    data, label = next(iter(test_loader))
    print("CIFAR10 TEST:", data.max(), data.min())

    train_loader, test_loader = get_loader("CIFAR100", 1)
    data, label = next(iter(train_loader))
    print("CIFAR100 TRAIN:", data.max(), data.min())
    data, label = next(iter(test_loader))
    print("CIFAR100 TEST:", data.max(), data.min())

    train_loader, test_loader = get_loader("imagenet", 1)
    data, label = next(iter(train_loader))
    print("IMAGENET TRAIN:", data.max(), data.min())
    data = data[0].permute(1, 2, 0) / 255
    plt.imshow(data)
    plt.title(imagenet_synset[label.item()])
    plt.show()

    data, label = next(iter(test_loader))
    print("IMAGENET TEST:", data.max(), data.min())
    data = data[0].permute(1, 2, 0) / 255
    plt.imshow(data)
    plt.title(imagenet_synset[label.item()])
    plt.show()
