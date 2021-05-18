"""
Load ccpd dataset.
"""
from torch.utils.data import DataLoader, Dataset
from os.path import join
import os
from .util import *
from PIL import Image


class CCPD(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        self.folders = [folder for folder in os.listdir(self.path) if folder.startswith("ccpd")]
        self.file_list = open(join(self.path, "splits", "train.txt" if train else "test.txt")).read().split()
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        pil_img = Image.open(join(self.path, file_name))
        if self.transform:
            pil_img = self.transform(pil_img)
        return pil_img, file_name


def get_ccpd(dataset_path, batch_size, num_workers):
    """
    return ccpd train & test dataloader
    :param dataset_path: dataset path
    :param batch_size: batch size
    :return: ccpd train & test dataloader
    """
    if num_workers is None:
        num_workers = 8
    train_set = CCPD(path=join(dataset_path, "CCPD2019"), train=True, transform=ccpd_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_set = CCPD(path=join(dataset_path, "CCPD2019"), train=False, transform=ccpd_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
