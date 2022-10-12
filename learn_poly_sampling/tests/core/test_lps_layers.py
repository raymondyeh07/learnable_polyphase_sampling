"""Implements test for LPS layer."""
import unittest
import torch
from layers.lps_utils import lps_downsample, lps_downsampleV2
from layers.lps_logit_layers import(LPSLogitLayersV1, LPSLogitLayersV2, SAInner,
                                    SAInner_bn, GraphLogitLayers, ComponentPerceptron)
from layers.polydown import split_polyV2


# Seed for reproducibility
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class TestLayerOpts(unittest.TestCase):
  def test_lps_layers_v1(self):
    batch_size, channel_size, height, width = 2, 3, 4, 4
    x = torch.randn(batch_size, channel_size, height, width)
    stride = 2
    model = LPSLogitLayersV1(3, 10, padding_mode='circular')
    poly_logits = model(x)
    # Circular Shift the input
    x_shift = torch.roll(x, shifts=(-1, -1), dims=(2, 3))
    poly_logits_shift = model(x_shift)
    poly_logits_shift_back = poly_logits_shift[:, [-1, -2, -3, -4]]
    # Check if shift results in permutation
    assert torch.allclose(poly_logits, poly_logits_shift_back)

  def test_lps_downsampling(self):
    batch_size, channel_size, height, width = 1, 1, 4, 4
    x = torch.randn(batch_size, channel_size, height, width)
    stride = 2
    # Toy permutation equivariant function, max element of the phase.
    xp0, _ = x[:, :, ::stride, ::stride].reshape(batch_size, -1).max(-1)
    xp1, _ = x[:, :, 1::stride, ::stride].reshape(batch_size, -1).max(-1)
    xp2, _ = x[:, :, ::stride, 1::stride].reshape(batch_size, -1).max(-1)
    xp3, _ = x[:, :, 1::stride, 1::stride].reshape(batch_size, -1).max(-1)
    poly_logits1 = torch.stack([xp0, xp1, xp2, xp3], dim=1)
    mode = 'test'
    # Set low temperature and hard samples approx. argmax.
    hard = True
    tau = 0.005
    out1 = lps_downsample(x, 2, poly_logits1, mode, hard=hard, tau=tau)
    # Circular Shift the input
    x_shift = torch.roll(x, shifts=(-1, -1), dims=(2, 3))
    xp0, _ = x_shift[:, :, ::stride, ::stride].reshape(batch_size, -1).max(-1)
    xp1, _ = x_shift[:, :, 1::stride, ::stride].reshape(batch_size, -1).max(-1)
    xp2, _ = x_shift[:, :, ::stride, 1::stride].reshape(batch_size, -1).max(-1)
    xp3, _ = x_shift[:, :, 1::stride, 1::stride].reshape(
        batch_size, -1).max(-1)
    poly_logits2 = torch.stack([xp0, xp1, xp2, xp3], dim=1)
    out2 = lps_downsample(x_shift, 2, poly_logits2, mode, hard=hard, tau=tau)
    # Check sum invariant at test time.
    try:
      assert torch.allclose(out1.mean((2, 3)), out2.mean((2, 3)))
    except AssertionError as e:
      print("[test_lps_downsampling] out1: ", out1)
      print("[test_lps_downsampling] out2: ", out2)
      raise e

  def test_lps_layers_v2(self):
    batch_size, channel_size, height, width = 3, 16, 8, 8
    x = torch.randn(batch_size, channel_size, height, width)
    stride = 2
    model = LPSLogitLayersV2(16, 32, padding_mode='circular')

    # Split polyphase
    num_components = stride ** 2
    components = split_polyV2(x=x,
                              stride=stride,
                              in_channels=channel_size,
                              num_components=num_components)
    poly_logits = model(components)

    # Circular shift
    x_shift = torch.roll(x, shifts=(-1, -1), dims=(2, 3))

    # Split polyphase
    components_shift = split_polyV2(x=x_shift,
                                    stride=stride,
                                    in_channels=channel_size,
                                    num_components=num_components)
    poly_logits_shift = model(components_shift)
    poly_logits_shift_back = poly_logits_shift[:, [-1, -2, -3, -4]]

    # Check results
    try:
      assert torch.allclose(poly_logits, poly_logits_shift_back)
    except AssertionError as e:
      print("[test_lps_layers_v2] poly_logits: ", poly_logits)
      print("[test_lps_layers_v2] poly_logits_shift: ", poly_logits_shift)
      raise e

  def test_lps_downsampling_V2(self):
    batch_size, channel_size, height, width = 3, 2, 4, 4
    x = torch.randn(batch_size, channel_size, height, width)
    stride = 2
    num_components = stride**2

    # Get phases
    components = split_polyV2(x=x,
                              stride=stride,
                              in_channels=channel_size,
                              num_components=num_components)

    # Toy permutation equivariant function, max element of the phase.
    xp0, _ = components[0].reshape(batch_size, -1).max(-1)
    xp1, _ = components[1].reshape(batch_size, -1).max(-1)
    xp2, _ = components[2].reshape(batch_size, -1).max(-1)
    xp3, _ = components[3].reshape(batch_size, -1).max(-1)
    poly_logits1 = torch.stack([xp0, xp1, xp2, xp3], dim=1)

    # Set low temperature and hard samples approx. argmax.
    mode = 'test'
    hard = True
    tau = 0.005
    out1, _ = lps_downsampleV2(
        components, 2, poly_logits1, mode, hard=hard, tau=tau)

    # Circular shift the input
    x_shift = torch.roll(x, shifts=(-1, -1), dims=(2, 3))
    components_shift = split_polyV2(x=x_shift,
                                    stride=stride,
                                    in_channels=channel_size,
                                    num_components=num_components)
    xp0, _ = components_shift[0].reshape(batch_size, -1).max(-1)
    xp1, _ = components_shift[1].reshape(batch_size, -1).max(-1)
    xp2, _ = components_shift[2].reshape(batch_size, -1).max(-1)
    xp3, _ = components_shift[3].reshape(batch_size, -1).max(-1)
    poly_logits2 = torch.stack([xp0, xp1, xp2, xp3], dim=1)

    out2, _ = lps_downsampleV2(
        components_shift, 2, poly_logits2, mode, hard=hard, tau=tau)

    # Check sum-shift invariant at test time.
    try:
      assert torch.allclose(out1.mean((2, 3)), out2.mean((2, 3)))
    except AssertionError as e:
      print("[test_lps_downsampling_v2] out1: ", out1)
      print("[test_lps_downsampling_v2] out2: ", out2)
      raise e

  def _test_logit_module(self, model, batch_size, channel_size, height, width, stride):
    model_name = type(model).__name__

    # Eval mode
    model.eval()

    x = torch.randn(batch_size, channel_size, height, width)
    # Polyphase rep
    num_components = stride**2
    components = split_polyV2(x=x,
                              stride=stride,
                              in_channels=channel_size,
                              num_components=num_components)
    poly_logits = model(components)

    # Circular shift
    x_shift = torch.roll(x, shifts=(1, 1), dims=(2, 3))

    # Shifted polyphase rep
    components_shift = split_polyV2(x=x_shift,
                                    stride=stride,
                                    in_channels=channel_size,
                                    num_components=num_components)
    poly_logits_shift = model(components_shift)
    poly_logits_shift_back = poly_logits_shift[:, [-1, -2, -3, -4]]

    # Check results
    try:
      assert torch.allclose(poly_logits, poly_logits_shift_back)
    except AssertionError as e:
      print("[test_%s] poly_logits: " % model_name, poly_logits)
      print("[test_%s] poly_logits_shift_back: " %
            model_name, poly_logits_shift_back)
      raise e

  def test_SAInner(self):
    batch_size, channel_size, height, width = 7, 3, 4, 4
    stride = 2
    model = SAInner(channel_size, channel_size, padding_mode='circular')
    self._test_logit_module(
        model, batch_size, channel_size, height, width, stride)

  def test_SAInner_bn(self):
    batch_size, channel_size, height, width = 7, 3, 4, 4
    stride = 2
    model = SAInner_bn(channel_size, channel_size, padding_mode='circular')
    self._test_logit_module(
        model, batch_size, channel_size, height, width, stride)

  def test_GraphLogitLayers(self):
    batch_size, channel_size, height, width = 7, 3, 4, 4
    stride = 2
    model = GraphLogitLayers(channel_size, channel_size, padding_mode='circular')
    self._test_logit_module(
        model, batch_size, channel_size, height, width, stride)

  def test_ComponentPerceptron(self):
    batch_size, channel_size, height, width = 7, 3, 4, 4
    stride = 2
    model = ComponentPerceptron(channel_size)
    self._test_logit_module(
        model, batch_size, channel_size, height, width, stride)


if __name__ == '__main__':
  unittest.main()
