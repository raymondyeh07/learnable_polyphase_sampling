"""Implements test for LPS layer."""
import unittest
import torch
import torch.nn as nn

from functools import partial
from layers.lps_utils import lps_downsample, lps_downsampleV2
from layers.lps_logit_layers import(LPSLogitLayersV1, LPSLogitLayersV2, SAInner,
                                    SAInner_bn)
from layers.polydown import split_polyV2, set_pool
from layers import(PolyphaseInvariantDown2D, max_p_norm, LPS,
                   get_logits_model, get_antialias)
from models.basic import BasicClassifier
from models.resnet_custom import ResNet18Custom


# Seed for reproducibility
import random
import numpy as np


class TestModelE2E(unittest.TestCase):
  def _test_model_circular_shift(self, model, model_name=None):
    if model_name is None:
      model_name = type(model).__name__

    # seed everything
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    # Eval mode: Switch to hard assoc.
    model.eval()
    batch_num = torch.randint(1, 7, (1,))
    x = torch.randn(batch_num, 3, 32, 32)
    random_shift = torch.randint(-3, 3, (2,))
    x_shift = torch.roll(x, shifts=(
        random_shift[0], random_shift[1]), dims=(2, 3))
    y = model(x)
    y_shift = model(x_shift)
    try:
      assert(torch.allclose(y, y_shift))
    except AssertionError as e:
      print('[test_%s y]: ' % model_name, y)
      print('[test_%s y_shift]: ' % model_name, y_shift)
      print('shifts:', random_shift)
      raise e

  def test_dummy_model(self):
    # Make Dummy Model.
    padding_mode = 'circular'
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=max_p_norm)
    padding = 1
    model = nn.Sequential(nn.Conv2d(3, 32, 3, 1, padding=padding, padding_mode=padding_mode),
                          set_pool(pooling_layer=pooling_layer, p_ch=32,h_ch=32),
                          torch.nn.AdaptiveAvgPool2d((1, 1)),
                          nn.Flatten(),
                          nn.Linear(32, 1)
                          )
    self._test_model_circular_shift(model, 'dummy_model')

  def test_basic_model(self):
    # Basic Model
    get_logits = get_logits_model('LPSLogitLayers')
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits)
    model = BasicClassifier(input_shape=(3, 32, 32),
                            num_classes=10,
                            padding_mode='circular',
                            pooling_layer=pooling_layer)
    self._test_model_circular_shift(model)

  def test_basic_model_LPSLogitLayersSkip(self):
    # Basic Model
    get_logits = get_logits_model('LPSLogitLayersSkip')
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits)
    model = BasicClassifier(input_shape=(3, 32, 32),
                            num_classes=10,
                            padding_mode='circular',
                            pooling_layer=pooling_layer)
    self._test_model_circular_shift(model)

  def test_basic_LPS_alias(self):
    # Basic (LPS) + LPF
    get_logits = get_logits_model('LPSLogitLayers')
    antialias_layer = get_antialias(antialias_mode='LowPassFilter',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=None)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits,
                            antialias_layer=antialias_layer)
    model = BasicClassifier(input_shape=(3, 32, 32),
                            num_classes=10,
                            padding_mode='circular',
                            pooling_layer=pooling_layer)
    self._test_model_circular_shift(model)

  def test_basic_LPS_DDAC(self):
    # Basic (LPS) + DDAC
    get_logits = get_logits_model('LPSLogitLayers')
    antialias_layer = get_antialias(antialias_mode='DDAC',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=2)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits,
                            antialias_layer=antialias_layer)
    model = BasicClassifier(input_shape=(3, 32, 32),
                            num_classes=10,
                            padding_mode='circular',
                            pooling_layer=pooling_layer)
    self._test_model_circular_shift(model)

  def test_basic_M2N_alias(self):
    # Basic (Max 2 norm) + LPF
    antialias_layer = get_antialias(antialias_mode='LowPassFilter',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=None)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=max_p_norm,
                            antialias_layer=antialias_layer)
    model = BasicClassifier(input_shape=(3, 32, 32),
                            num_classes=10,
                            padding_mode='circular',
                            pooling_layer=pooling_layer)
    self._test_model_circular_shift(model)

  def test_basic_M2N_DDAC(self):
    # Basic (Max 2 norm) + DDAC
    antialias_layer = get_antialias(antialias_mode='DDAC',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=2)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=max_p_norm,
                            antialias_layer=antialias_layer)
    model = BasicClassifier(input_shape=(3, 32, 32),
                            num_classes=10,
                            padding_mode='circular',
                            pooling_layer=pooling_layer)
    self._test_model_circular_shift(model)

  def test_resnet_model(self):
    # ResNet18Custom Model
    get_logits = get_logits_model('LPSLogitLayers')
    # Extra model args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': False,
        'apply_maxpool': False,
        'ret_prob': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits)
    model = ResNet18Custom(input_shape=(3, 32, 32),
                           num_classes=10,
                           padding_mode='circular',
                           pooling_layer=pooling_layer,
                           extras_model=extras_model)
    self._test_model_circular_shift(model)

  def test_resnet_LPS_alias(self):
    # ResNet18Custom (LPS) + LPF
    get_logits = get_logits_model('LPSLogitLayers')
    # Extra model args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': False,
        'apply_maxpool': False,
        'ret_prob': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }
    antialias_layer = get_antialias(antialias_mode='LowPassFilter',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=None)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits,
                            antialias_layer=antialias_layer)
    model = ResNet18Custom(input_shape=(3, 32, 32),
                           num_classes=10,
                           padding_mode='circular',
                           pooling_layer=pooling_layer,
                           extras_model=extras_model)
    self._test_model_circular_shift(model)

  def test_resnet_LPS_DDAC(self):
    # ResNet18Custom (LPS) + DDAC
    get_logits = get_logits_model('LPSLogitLayers')
    # Extra model args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': False,
        'apply_maxpool': False,
        'ret_prob': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }
    antialias_layer = get_antialias(antialias_mode='DDAC',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=2)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits,
                            antialias_layer=antialias_layer)
    model = ResNet18Custom(input_shape=(3, 32, 32),
                           num_classes=10,
                           padding_mode='circular',
                           pooling_layer=pooling_layer,
                           extras_model=extras_model)
    self._test_model_circular_shift(model)

  def test_resnet_M2N_alias(self):
    # ResNet18Custom (Max 2 norm) + LPF
    # Extra model args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': False,
        'apply_maxpool': False,
        'ret_prob': True,
'ret_logits': False,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }
    antialias_layer = get_antialias(antialias_mode='LowPassFilter',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=None)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=max_p_norm,
                            antialias_layer=antialias_layer)
    model = ResNet18Custom(input_shape=(3, 32, 32),
                           num_classes=10,
                           padding_mode='circular',
                           pooling_layer=pooling_layer,
                           extras_model=extras_model)
    self._test_model_circular_shift(model)

  def test_resnet_M2N_DDAC(self):
    # ResNet18Custom (Max 2 norm) + DDAC
    # Extra model args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': False,
        'apply_maxpool': False,
        'ret_prob': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }
    antialias_layer = get_antialias(antialias_mode='DDAC',
                                    antialias_size=7,
                                    antialias_padding='same',
                                    antialias_padding_mode='circular',
                                    antialias_group=2)
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=max_p_norm,
                            antialias_layer=antialias_layer)
    model = ResNet18Custom(input_shape=(3, 32, 32),
                           num_classes=10,
                           padding_mode='circular',
                           pooling_layer=pooling_layer,
                           extras_model=extras_model)
    self._test_model_circular_shift(model)

if __name__ == '__main__':
  unittest.main()
