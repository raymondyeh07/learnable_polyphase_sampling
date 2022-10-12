import os
from copy import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms

from .custom_transforms import RandomCircularShift

import pytorch_lightning as pl

CPU_COUNT = os.cpu_count()


class CIFAR10DataModule(pl.LightningDataModule):
  def __init__(self, batch_size, data_dir: str = './', aug_method='basic',
               base_center_crop=32, max_shift=3, val_split=0.1):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.val_split = val_split

    cifar_mean = torch.tensor([0.49141738, 0.48219556, 0.44662726])
    cifar_std = torch.tensor([0.24703224, 0.24348514, 0.26158786])

    assert aug_method in ['basic', 'shift', 'circular',
                          'randomresized']

    self.train_transform = [transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(),
                            transforms.Normalize(cifar_mean, cifar_std)]
    self.test_transform = [transforms.ToTensor(),
                           transforms.Normalize(cifar_mean, cifar_std)]
    if aug_method == 'shift':
      self.train_transform.insert(0, transforms.RandomCrop(
          base_center_crop, padding=max_shift))
      self.test_transform.insert(0, transforms.Pad(max_shift))
      self.test_transform.insert(
          1, transforms.CenterCrop(size=base_center_crop))
    elif aug_method == 'circular':
      self.train_transform.insert(
          0, transforms.CenterCrop(size=base_center_crop))
      self.train_transform.append(RandomCircularShift(max_shift))
      self.test_transform.insert(
          0, transforms.CenterCrop(size=base_center_crop))
    elif aug_method == 'randomresized':
      # Insert RandomResizedCrop
      self.train_transform.insert(1, transforms.RandomResizedCrop(base_center_crop))
    self.train_transform = transforms.Compose(self.train_transform)
    self.test_transform = transforms.Compose(self.test_transform) 

    self.dims = (3, 32, 32)
    self.num_classes = 10

  def prepare_data(self):
    # download
    CIFAR10(self.data_dir, train=True, download=True)
    CIFAR10(self.data_dir, train=False, download=True)

  def setup(self, stage=None):
    # Assign train/val datasets for use in dataloaders
    if stage == 'fit' or stage is None:
      if self.val_split is None:
        # Don't split trainset
        self.cifar_train = CIFAR10(self.data_dir,
                                   train=True,
                                   transform=self.train_transform)
        # Don't apply augmentation to val. set.
        self.cifar_val = CIFAR10(self.data_dir,
                                 train=False,
                                 transform=self.test_transform)
      else:
        # Split trainset
        cifar_full = CIFAR10(self.data_dir,
                             train=True,
                             transform=self.train_transform)
        full_len = len(cifar_full)
        val_len = int(full_len*self.val_split)
        train_len = full_len-val_len
        self.cifar_train, self.cifar_val = random_split(
            cifar_full, [train_len, val_len])
        # Don't apply augmentation to val. set.
        self.cifar_val.dataset = copy(cifar_full)
        self.cifar_val.dataset.transform = self.test_transform

    # Assign test dataset for use in dataloader(s)
    if stage == 'test' or stage is None:
      self.cifar_test = CIFAR10(
          self.data_dir, train=False, transform=self.test_transform)

  def train_dataloader(self):
    return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=CPU_COUNT)

  def val_dataloader(self):
    return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=CPU_COUNT)

  def test_dataloader(self):
    return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=CPU_COUNT)
