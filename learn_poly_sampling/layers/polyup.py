import torch
import torch.nn as nn
import numpy as np
from .lps_utils import lps_upsampleV2


def unpool_multistage(x,x_layer,p_dict,
                      scale_factor,unpool_layer):
  # Refs lookup table
  assert np.log2(scale_factor)%1==0,("Scale factor not a power of 2. "\
                                     "Only pool factors of 2 is currently supported")
  assert x_layer in ['low_level','out']
  if x_layer=='low_level':
    _layers = ['maxpool_3','maxpool_0']
  elif x_layer=='out':
    _layers = ['out','layer3','layer2']

  # Select refs based on scale factor
  steps = int(np.log2(scale_factor))
  assert steps<=len(_layers),\
    "Steps exceed number of unpooling indices. Check 'scale_factor' argument."
  layers = _layers[:steps]

  # Unpool
  for i in range(len(layers)):
    layer = layers[i]
    x = unpool_layer[i](x=x,prob=p_dict[layer])
  return x
    

# Set unpool
def set_unpool(unpool_layer,p_ch,no_antialias=False):
  if unpool_layer.func.__name__=="PolyphaseInvariantUp2D":
    # Check valid functions
    assert unpool_layer.keywords["component_selection"].__name__ in ["LPS_u","max_p_norm_u"]
    if unpool_layer.keywords["component_selection"].__name__=="LPS_u":
      u=unpool_layer(stride=2,
                     in_channels=p_ch,
                     hid_channels=p_ch,
                     pass_extras=True,
                     no_antialias=no_antialias)
    elif unpool_layer.keywords["component_selection"].__name__=="max_p_norm_u":
      # in_channels required for antialias
      u=unpool_layer(stride=2,
                     in_channels=p_ch,
                     no_antialias=no_antialias)
  else:
    raise ValueError("Undefined unpooling layer. Check 'unpool_method' input argument.")
  return u


# Max p norm unpooling
def max_p_norm_u(x,prob,stride=2):
  """ M2N Upsampling Layer
  Args:
      1. x ([tensor]): Batch Size x Channel x Height x Width.
      2. prob (max_indices) ([tensor]): Batch Size x 1.
  """
  assert stride == 2  # TODO: Implement support for different stride factors.
  bb,cc,hh,ww = x.shape
  x_poly = torch.repeat_interleave(x,stride**2,dim=1)
  hard_prob = torch.nn.functional.one_hot(prob,stride**2)
  hard_prob_poly = hard_prob.repeat([1,cc]).view(bb,cc*stride**2,1,1)
  x_poly_weighted = x_poly*hard_prob_poly
  x_up = nn.functional.pixel_shuffle(x_poly_weighted,stride)
  return x_up,prob


# LPS unpool selection
class LPS_u(nn.Module):
  def __init__(self,in_channels,hid_channels,
               stride,upsample=lps_upsampleV2,
               get_samples=False):
    super(LPS_u,self).__init__()

    self.upsample = upsample
    self.stride = stride
    self.get_samples = get_samples
    tau = torch.tensor(1.0,dtype=torch.float32)
    self.register_buffer('gumbel_tau',tau)

  def forward(self,x,prob,
              mode="train"):
    # Pass pre-computed probability/logits. Output
    # upscaled phase and polyphase probability
    out,_prob = self.upsample(x=x,
                              stride=self.stride,
                              tau=self.gumbel_tau,
                              get_samples=self.get_samples,
                              mode=mode,
                              prob=prob)
    return out,_prob


# Polyphase unpooling layer
class PolyphaseInvariantUp2D(nn.Module):
  def __init__(self,component_selection,get_samples=None,
               comp_convex=False,stride:int=2,pass_extras=False,
               in_channels=None,hid_channels=None,antialias_layer=None,
               no_antialias=False):
    super().__init__()
    assert stride==2
    self.pass_extras = pass_extras
    self.no_antialias = no_antialias
    self.comp_convex = comp_convex
    if self.no_antialias: self.antialias_layer = None
    else: self.antialias_layer = antialias_layer

    if self.pass_extras:
      # LPS: Pass extra pars
      self.component_selection = component_selection(in_channels=in_channels,
                                                     hid_channels=hid_channels,
                                                     stride=stride,
                                                     get_samples=get_samples,
                                                     )
    else:
      self.component_selection = component_selection

    if self.antialias_layer is not None:
      # Antialiasing layer
      self._antialias_layer = self.antialias_layer(in_channels=in_channels)

  def forward(self,x,ret_prob=False,
              prob=None):
    # Unpool
    if self.pass_extras:
      # LPS_u: set gumbel-softmax mode.
      if self.training:
        # Note that gumbel-softmax training mode
        # is already passed via get_samples
        component_mode = "train"
      else:
        # Switch to convex combination if required
        component_mode = "test_convex" if self.comp_convex else "test"

      # Pass pre-computed prob/logits, components and gumbel-smax mode
      # Return optimal phase and polyphase prob/logits
      _x,_p = self.component_selection(x=x,
                                       mode=component_mode,
                                       prob=prob)       
    else:
      # Max_p_norm_u
      # Pass pre-computed max index and components
      _x,_p = self.component_selection(x=x,
                                       prob=prob)

    if self.antialias_layer is not None:
      # Apply antialias
      _x = self._antialias_layer(_x)

    # Return fmap and logits
    if ret_prob: return _x,_p
    else: return _x

