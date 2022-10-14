import unittest
from layers import PolyphaseInvariantDown2D, max_p_norm
from layers.polydown import split_polyV1, split_polyV2
import torch


# Seed for reproducibility
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class TestPolyphaseInvariantDown2D(unittest.TestCase):
  def test_polydown(self):
    x = torch.randn(32, 512, 32, 32)
    poly_down = PolyphaseInvariantDown2D(2)
    y = poly_down(x, prob=0)

    assert y.shape == torch.Size([32, 512, 16, 16])

  def test_polydown_max_pnorm(self):
    x = torch.randn(32, 512, 32, 32)
    poly_down = PolyphaseInvariantDown2D(2, max_p_norm)
    y = poly_down(x)

    assert y.shape == torch.Size([32, 512, 16, 16])

  # Check phase order
  def test_polydown_split_poly(self):
    stride = 2
    errtol = 1e-6
    batch_size, channel_size, height, width = 1, 2, 4, 4
    x = torch.arange(batch_size*channel_size*height*width).reshape(
        batch_size, channel_size, height, width).float()
    #x= torch.randn( batch_size, channel_size, height, width)
    num_components = stride ** 2

    # Target split
    xpoly_0 = x[:, :, ::stride, ::stride]
    xpoly_1 = x[:, :, ::stride, 1::stride]
    xpoly_2 = x[:, :, 1::stride, ::stride]
    xpoly_3 = x[:, :, 1::stride, 1::stride]
    xpoly_target = torch.cat([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim=0)

    # Previous split
    compsV1 = split_polyV1(x=x,
                           stride=stride,
                           in_channels=channel_size,
                           num_components=num_components)
    xpolyV1 = torch.cat(compsV1, dim=0)

    # Current split
    compsV2 = split_polyV2(x=x,
                           stride=stride,
                           in_channels=channel_size,
                           num_components=num_components)
    xpolyV2 = torch.flatten(compsV2,start_dim=0, end_dim=1)

    # Compute error
    errvalV1 = torch.norm(xpoly_target - xpolyV1)
    errvalV2 = torch.norm(xpoly_target - xpolyV2)

    # Check results
    try:
      assert(errvalV2 < errtol)
      #assert( errvalV1< errtol)
    except AssertionError as e:
      print("errvalV1: ", errvalV1)
      print("errvalV2: ", errvalV2)
      raise e
 


if __name__ == '__main__':
  unittest.main()
