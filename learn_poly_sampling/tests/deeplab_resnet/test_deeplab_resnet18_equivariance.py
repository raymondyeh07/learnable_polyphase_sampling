import torch
import random
import unittest
import numpy as np
from layers import(PolyphaseInvariantDown2D,LPS,get_logits_model,
                   get_antialias)
from layers.polyup import(PolyphaseInvariantUp2D,LPS_u)
from models.base_segmentation import DDAC_MODEL_MAP, DDACSegmentation
from functools import partial
from models import get_model as get_backbone


class TestSegEquivariance(unittest.TestCase):
  # DDACSegmentation + unpooling
  def test_deeplab_resnet18_unpool(self):
    # seed everything
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    # Input
    b,c,h,w = 4,3,512,512
    s = [-1,-1]
    x = torch.randn(b,c,h,w).float()
    x_shift = torch.roll(x,shifts=(s[0],s[1]),dims=(2,3))

    # Pooling layer
    get_logits = get_logits_model('LPSLogitLayers')
    pool_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits,
                            antialias_layer=None)

    # Unpool layer
    unpool_layer = partial(PolyphaseInvariantUp2D,
                           component_selection=LPS_u,
                           antialias_layer=None)

    # Extra backbone args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': True,
        'apply_maxpool': True,
        'ret_prob': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }

    # Backbone
    backbone = get_backbone('ResNet18Custom')(
      input_shape=(),
      num_classes=21,
      padding_mode='circular',
      pooling_layer=pool_layer,
      extras_model=extras_model)

    # Segmenter
    model = DDACSegmentation(
        model_name='deeplabv3plus_resnet_lps_unpool',
        num_classes=21,
        output_stride=16,
        backbone=backbone,
        unpool_layer=unpool_layer,
        classifier_padding_mode='circular')
    model.eval()

    with torch.no_grad():
      # Output
      y = model(x)
      y_shift = model(x_shift)

      # Reference
      _y_shift = torch.roll(y,shifts=(s[0],s[1]),dims=(2,3))

      # Check error
      errval = torch.norm(_y_shift-y_shift)
    print("[test_deeplab_resnet18_unpool] errval: ",errval)
    assert torch.allclose(y_shift,_y_shift)

  # DDACSegmentation + unpooling + antialias
  def test_deeplab_resnet18_unpool_lpf(self):
    # seed everything
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Input
    b,c,h,w = 4,3,512,512
    s = [-1,-1]
    x = torch.randn(b,c,h,w).float()
    x_shift = torch.roll(x,shifts=(s[0],s[1]),dims=(2,3))

    # Antialias pars
    antialias_mode = 'LowPassFilter'
    antialias_size = 3
    antialias_padding = 'same'
    antialias_padding_mode = 'circular'
    antialias_group = None
    unpool_antialias_scale = 2

    # Antialias filters
    antialias = get_antialias(antialias_mode=antialias_mode,
                              antialias_size=antialias_size,
                              antialias_padding=antialias_padding,
                              antialias_padding_mode=antialias_padding_mode,
                              antialias_group=antialias_group)
    unpool_antialias = get_antialias(antialias_mode=antialias_mode,
                                     antialias_size=antialias_size,
                                     antialias_padding=antialias_padding,
                                     antialias_padding_mode=antialias_padding_mode,
                                     antialias_group=antialias_group,
                                     antialias_scale=unpool_antialias_scale)

    # Pooling layer
    get_logits = get_logits_model('LPSLogitLayers')
    pool_layer = partial(PolyphaseInvariantDown2D,
                         component_selection=LPS,
                         get_logits=get_logits,
                         antialias_layer=antialias)

    # Unpool layer
    unpool_layer = partial(PolyphaseInvariantUp2D,
                           component_selection=LPS_u,
                           antialias_layer=unpool_antialias)

    # Extra backbone args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': True,
        'apply_maxpool': True,
        'ret_prob': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }

    # Backbone
    backbone = get_backbone('ResNet18Custom')(
      input_shape=(),
      num_classes=21,
      padding_mode='circular',
      pooling_layer=pool_layer,
      extras_model=extras_model)

    # Segmenter
    model = DDACSegmentation(
        model_name='deeplabv3plus_resnet_lps_unpool',
        num_classes=21,
        output_stride=16,
        backbone=backbone,
        unpool_layer=unpool_layer,
        classifier_padding_mode='circular')
    model.eval()

    with torch.no_grad():
      # Output
      y = model(x)
      y_shift = model(x_shift)

      # Reference
      _y_shift = torch.roll(y,shifts=(s[0],s[1]),dims=(2,3))

      # Check error
      errval = torch.norm(_y_shift-y_shift)
    print("[test_deeplab_resnet18_unpool_lpf] errval: ",errval)
    assert torch.allclose(y_shift,_y_shift)

  # DDACSegmentation + unpooling + DDAC
  def test_deeplab_resnet18_unpool_ddac(self):
    # seed everything
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Input
    b,c,h,w = 4,3,512,512
    s = [-1,-1]
    x = torch.randn(b,c,h,w).float()
    x_shift = torch.roll(x,shifts=(s[0],s[1]),dims=(2,3))

    # Antialias pars
    antialias_mode = 'DDAC'
    antialias_size = 3
    antialias_padding = 'same'
    antialias_padding_mode = 'circular'
    antialias_group = 1
    unpool_antialias_scale = 2

    # Antialias filters
    antialias = get_antialias(antialias_mode=antialias_mode,
                              antialias_size=antialias_size,
                              antialias_padding=antialias_padding,
                              antialias_padding_mode=antialias_padding_mode,
                              antialias_group=antialias_group)
    unpool_antialias = get_antialias(antialias_mode=antialias_mode,
                                     antialias_size=antialias_size,
                                     antialias_padding=antialias_padding,
                                     antialias_padding_mode=antialias_padding_mode,
                                     antialias_group=antialias_group,
                                     antialias_scale=unpool_antialias_scale)

    # Pooling layer
    get_logits = get_logits_model('LPSLogitLayers')
    pool_layer = partial(PolyphaseInvariantDown2D,
                         component_selection=LPS,
                         get_logits=get_logits,
                         antialias_layer=antialias)

    # Unpool layer
    unpool_layer = partial(PolyphaseInvariantUp2D,
                           component_selection=LPS_u,
                           antialias_layer=unpool_antialias)

    # Extra backbone args
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': True,
        'apply_maxpool': True,
        'ret_prob': True,
        'forward_pool_method': 'LPS',
        'forward_ret_prob_logits': False,
    }

    # Backbone
    backbone = get_backbone('ResNet18Custom')(
      input_shape=(),
      num_classes=21,
      padding_mode='circular',
      pooling_layer=pool_layer,
      extras_model=extras_model)

    # Segmenter
    model = DDACSegmentation(
        model_name='deeplabv3plus_resnet_lps_unpool',
        num_classes=21,
        output_stride=16,
        backbone=backbone,
        unpool_layer=unpool_layer,
        classifier_padding_mode='circular')
    model.eval()

    with torch.no_grad():
      # Output
      y = model(x)
      y_shift = model(x_shift)

      # Reference
      _y_shift = torch.roll(y,shifts=(s[0],s[1]),dims=(2,3))

      # Check error
      errval = torch.norm(_y_shift-y_shift)
    print("[test_deeplab_resnet18_unpool_ddac] errval: ",errval)
    assert torch.allclose(y_shift,_y_shift)


if __name__ == '__main__':
  unittest.main()
