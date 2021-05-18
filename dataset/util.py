"""
Some assistant functions.
"""
from torchvision.transforms import *


class Float2Int(object):
    """
    A transformer, convert floating pixels to integer pixels
    By multiplying 255.
    In integer inference, this is necessary.
    """

    def __call__(self, tensor):
        return tensor * 255

    def __repr__(self):
        return self.__class__.__name__ + '()'


cifar_train_transform = Compose([
    RandomCrop([28, 28]),
    RandomHorizontalFlip(),
    Resize([32, 32]),
    RandomRotation(20),
    ToTensor(),
    Float2Int()
])

cifar_test_transform = Compose([
    ToTensor(),
    Float2Int()
])

ccpd_transform = Compose([
    CenterCrop([700, 700]),
    Resize([256, 256]),
    ToTensor(),
    Float2Int()
])
