"""Test for Replacing Circular Convolution"""
import unittest
import random
import torch as ch
import torch.nn as nn

# Seed for reproducibility
import random
import numpy as np

random.seed(0)
np.random.seed(0)
ch.manual_seed(0)
class TestCircularPad(unittest.TestCase):
  def test_output_size(self):
    # TODO: Replace by checking Pytorch version. Old versions have a bug on
    # the circular padding implementation, creating an output size inconsistency.
    # https://github.com/pytorch/pytorch/issues/20981

    # Input
    b,c,h,w=8,3,32,32
    x=ch.randn(b,c,h,w)

    # Conv layers
    _k=random.randint(1,5)
    k=2*_k+1
    h_z=nn.Conv2d(c,c,k,padding=_k,padding_mode="zeros")
    h_c=nn.Conv2d(c,c,k,padding=_k,padding_mode="circular")

    # Output
    y_z=h_z(x)
    y_c=h_c(x)

    assert y_z.shape==y_c.shape

  def test_padding_attribute(self):
    # Check replacing padding_mode on-the-fly
    # is equivalent to declarating it initially.

    # Input
    b,c,h,w=1,1,32,32
    x=ch.randn(b,c,h,w)

    # Conv layers w. same weights
    _k=random.randint(1,5)
    k=2*_k+1
    h_z=nn.Conv2d(c,c,k,padding=_k,padding_mode="zeros")
    h_c=nn.Conv2d(c,c,k,padding=_k,padding_mode="circular")
    h_c.weight=nn.Parameter(h_z.weight.clone())
    h_c.bias=nn.Parameter(h_z.bias.clone())


    # Original output
    y_z=h_z(x)
    y_c=h_c(x)

    # Modify pad to circular
    h_z.padding_mode="circular"
    y_cc=h_z(x)

    # Check error
    err_z=ch.norm(y_z-y_c)
    err_cc=ch.norm(y_cc-y_c)

    try:
      assert ch.allclose(y_cc,y_c)
    except AssertionError as e:
      print("[test_padding_attribute] err_z: ", err_z)
      print("[test_padding_attribute] err_cc: ", err_cc)
      print('shifts:', random_shift)
      raise e
