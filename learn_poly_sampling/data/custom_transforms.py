"""Implements custom transforms."""

import torch
import random


class RandomCircularShift(object):
  def __init__(self, max_shift):
    self.max_shift = max_shift

  def get_params(self):
    i = random.randint(-self.max_shift, self.max_shift)
    j = random.randint(-self.max_shift, self.max_shift)
    return i, j

  def __call__(self, img):
    i, j = self.get_params()
    return torch.roll(img, shifts=(i, j), dims=(1, 2))


class RandomCircularShiftWithIndex(object):
  def __init__(self, max_shift):
    self.max_shift = max_shift

  def get_params(self):
    i = random.randint(-self.max_shift, self.max_shift)
    j = random.randint(-self.max_shift, self.max_shift)
    return i, j

  def __call__(self, img):
    i, j = self.get_params()
    return torch.roll(img, shifts=(i, j), dims=(1, 2))


class RandomCropWithIndex(object):
  def __init__(self, max_shift, crop_size=513):
      self.max_shift = max_shift
      self.crop_size = crop_size

  def get_params(self):
      i = random.randint(0, self.max_shift)
      j = random.randint(0, self.max_shift)
      return i, j

  def __call__(self, img):
      i, j = self.get_params()
      return img[:, i:i+self.crop_size, j:j+self.crop_size]
