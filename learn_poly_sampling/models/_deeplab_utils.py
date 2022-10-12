import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from layers.polyup import set_unpool,unpool_multistage
from .thirdparty.network.utils import IntermediateLayerGetter


class IntermediateLayerGetterUnpool(IntermediateLayerGetter):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers,
                 unpool_layer=None):
        super().__init__(model=model,
                         return_layers=return_layers)
        self.unpool_layer = unpool_layer

    def forward_unpool(self,x,out,
                       p_dict):
      for name, module in self.named_children():
        p = None
        if name=="maxpool":
          # Keep maxpool feats and prob
          _n = self.return_layers[name] if name in self.return_layers else name
          s = len(module)
          for i,layer in zip(range(s),module):
            _layer = layer.__class__.__name__
            if _layer == "PolyphaseInvariantDown2D":
              # Save feats and prob
              x,p = layer(x=x,ret_prob=True)
              _name = _n + '_' + str(i)
              out[_name] = x
              if p is not None: p_dict[_name] = p
            else:
              # Save feats only
              x = layer(x)
        elif name in ['layer2','layer3','layer4']:
          # Keep BasicBlock/Bottleneck feats and prob
          # Note: layer 1 has no pool
          _name = self.return_layers[name] if name in self.return_layers else name
          x,p = module[0](x=x,global_ret_prob=True)
          for i in range(1,len(module)): x = module[i](x)

          # Save feats and prob
          out[_name] = x
          if p is not None: p_dict[_name] = p
        else:
          # Compute and save feats
          _name = self.return_layers[name] if name in self.return_layers else name
          x = module(x)
          out[_name] = x
          if p is not None: p_dict[_name] = p
      return out,p_dict

    def forward(self, x):
        out = OrderedDict()
        if self.unpool_layer is not None:
          # Compute feats and prob
          p_dict = OrderedDict()
          self.forward_unpool(x=x,
                              out=out,
                              p_dict=p_dict)
          return out,p_dict
        else:
          # Output feats 
          for name, module in self.named_children():
              x = module(x)
              if name in self.return_layers:
                  out_name = self.return_layers[name]
                  out[out_name] = x
          return out

