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
from models.resnet_custom import ResNet50Custom


class TestResNet50Grad(unittest.TestCase):
  def _test_model_grad(self, model, model_name=None):
    if model_name is None:
      model_name = type(model).__name__
    # Eval mode: Switch to hard assoc.
    model.train()
    batch_num = torch.randint(1, 7, (1,))
    x = torch.randn(batch_num, 3, 224, 224)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Grad
    none_grad_list = []
    for key,param in model.named_parameters():
      if param.grad is None:
        none_grad_list.append(key)

    try:
      # Check if empty list
      assert(not(none_grad_list))
    except AssertionError as e:
      print('[ResNet50Grad] none_grad_list:', none_grad_list)
      raise e

  def test_resnet50_grad(self):
    # ResNet50Custom Model
    get_logits = get_logits_model('LPSLogitLayers')
    # Extra model args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': True,
        'apply_maxpool': True,
        'ret_prob': True,
        'ret_logits': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits)
    model = ResNet50Custom(input_shape=(3, 224, 224),
                           num_classes=10,
                           padding_mode='circular',
                           pooling_layer=pooling_layer,
                           extras_model=extras_model
                           )
    self._test_model_grad(model)

if __name__ == '__main__':
  unittest.main()
