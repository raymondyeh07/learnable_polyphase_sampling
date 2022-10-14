import os
from copy import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms

from .custom_transforms import RandomCircularShift

import pytorch_lightning as pl

CPU_COUNT = os.cpu_count()


class ImageNetDataModule(pl.LightningDataModule):
  def __init__(self, batch_size, data_dir: str = './', aug_method='basic',
               base_center_crop=224, max_shift=32, val_split=None):
    super().__init__()
    self.data_dir = data_dir
    self.train_dir= os.path.join(data_dir,'train')
    self.test_dir= os.path.join(data_dir,'val')
    self.batch_size = batch_size
    self.val_split = val_split
    self.dims = (3, 224, 224)
    self.num_classes = 1000
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
    imagenet_std = torch.tensor([0.229, 0.224, 0.225])

    # Base transformations
    assert aug_method in ['basic', 'shift', 'circular',
                          'randomresized', 'shift_consistency']
    self.train_transform = [transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(imagenet_mean,imagenet_std)]
    self.test_transform = [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(imagenet_mean,imagenet_std)]

    # Data aug
    if aug_method == 'shift':
      self.train_transform[2] = transforms.RandomCrop(base_center_crop,
                                                      padding=max_shift)
      self.test_transform.insert(0, transforms.Pad(max_shift))
      self.test_transform.insert(
          1, transforms.CenterCrop(size=base_center_crop))
    elif aug_method == 'circular':
      # CenterCrop already in default transf.
      self.train_transform.append(RandomCircularShift(max_shift))
    elif aug_method == 'randomresized':
      # Replace CenterCrop by RandomResizedCrop
      del self.train_transform[0]
      self.train_transform[0] = transforms.RandomResizedCrop(base_center_crop)
    elif aug_method == 'shift_consistency':
      # Increase crop dims to allow shifts
      self.test_transform[1] = transforms.CenterCrop(base_center_crop)
    self.train_transform = transforms.Compose(self.train_transform)
    self.test_transform = transforms.Compose(self.test_transform)

  def setup(self, stage=None):
    # Assign train/val datasets for use in dataloaders
    if stage == 'fit' or stage is None:
      if self.val_split is None:
        # Don't split trainset
        self.imagenet_train = ImageFolder(self.train_dir,
                                          transform=self.train_transform)
        # Don't apply augmentation to val. set.
        self.imagenet_val = ImageFolder(self.test_dir,
                                        transform=self.test_transform)
      else:
        # Split trainset
        imagenet_full = ImageFolder(self.train_dir,
                                    transform=self.train_transform)
        full_len = len(imagenet_full)
        val_len = int(full_len*self.val_split)
        train_len = full_len-val_len
        self.imagenet_train, self.imagenet_val = random_split(
            imagenet_full, [train_len, val_len])
        # Don't apply augmentation to val. set.
        self.imagenet_val.dataset = copy(imagenet_full)
        self.imagenet_val.dataset.transform = self.test_transform

    # Assign test dataset for use in dataloader(s)
    if stage == 'test' or stage is None:
      self.imagenet_test = ImageFolder(self.test_dir,
                                       transform=self.test_transform)

  def train_dataloader(self):
    return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True, num_workers=CPU_COUNT)

  def val_dataloader(self):
    return DataLoader(self.imagenet_val, batch_size=self.batch_size, num_workers=CPU_COUNT)

  def test_dataloader(self):
    return DataLoader(self.imagenet_test, batch_size=self.batch_size, num_workers=CPU_COUNT)
