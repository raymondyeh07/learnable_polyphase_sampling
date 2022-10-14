import torch
from torch import nn
from layers.lps_logit_layers import LPSLogitLayersV2
from layers.lps_utils import lps_downsampleV2
from layers.lowpass_filter import LowPassFilter,DDAC


# Set pool
def set_pool(pooling_layer,p_ch,h_ch=None,
             use_get_logits=True,no_antialias=False):
  if pooling_layer.func.__name__=="PolyphaseInvariantDown2D":
    # Check valid functions
    assert pooling_layer.keywords["component_selection"].__name__ in ["LPS","max_p_norm"]
    if pooling_layer.keywords["component_selection"].__name__=="LPS":
      p=pooling_layer(stride=2,
                      in_channels=p_ch,
                      hid_channels=h_ch,
                      pass_extras=True,
                      use_get_logits=use_get_logits,
                      no_antialias=no_antialias)
    elif pooling_layer.keywords["component_selection"].__name__=="max_p_norm":
      # in_channels required for antialias
      p=pooling_layer(stride=2,
                      in_channels=p_ch,
                      use_get_logits=use_get_logits,
                      no_antialias=no_antialias)
  elif pooling_layer.func.__name__=="Decimation":
    # Antialias + downsampling. Antialias and stride in partial.
    # in_channels required for antialias
    p=pooling_layer(in_channels=p_ch,
                    no_antialias=no_antialias)
  elif pooling_layer.func.__name__=="AvgPool2d":
    # Stride included in partial
    p=pooling_layer()
  else:
    raise ValueError("Undefined pooling layer. Check 'pool_method' input argument.")
  return p


# Dummy for sanity checks
def fixed_component(x, prob,x_nofilt=None):
  assert prob is not None

  # Stack components
  #_x = torch.stack(x, dim=4)
  #_x = x.permute(1,2,3,4,0)
  #_x = _x[torch.arange(_x.shape[0]), :, :, :, prob]
  _x = x[prob, torch.arange(x.shape[1]), :, :, :]
  # return index for consistency purposes
  return _x, prob


def max_p_norm(x, p=2, prob=None,
               x_nofilt=None):
  if prob is None:
    # Compute max index
    if x_nofilt is not None:
      # Compute indices from unfiltered components
      norms = [torch.norm(c.reshape(c.shape[0], -1), p=p, dim=1) for c in x_nofilt]
    else:
      norms = [torch.norm(c.reshape(c.shape[0], -1), p=p, dim=1) for c in x]
    norms = torch.stack(norms, dim=1)
    idx = torch.argmax(norms, dim=1)
  else:
    # Use precomputed max index (prob)
    idx= prob
  _x = x[idx, torch.arange(x.shape[1]), :, :, :]
  # Output fmap and index
  return _x, idx


# LPS selection
class LPS(nn.Module):
  def __init__(self, in_channels, hid_channels,
               stride, logits_pad, get_logits=LPSLogitLayersV2,
               downsample=lps_downsampleV2):
    super(LPS, self).__init__()

    if get_logits is not None:
     # If not None, compute logits
     # Else, LPS receives precomputed indices.
     self.get_logits = get_logits(in_channels=in_channels,
                                  hid_channels=hid_channels,
                                  padding_mode=logits_pad)
    self.downsample = downsample
    self.stride = stride

    tau = torch.tensor(1.0, dtype=torch.float32)
    self.register_buffer('gumbel_tau', tau)

  def forward(self,x,mode="train",
              prob=None,x_nofilt=None):
    if prob is None:
        # Get logits (CNN-based)
        if x_nofilt is not None:
          # Compute logits from unfiltered components
          logits = self.get_logits(x_nofilt)
        else:
          logits = self.get_logits(x)

        # Output selected phase and polyphase probability
        out, _prob = self.downsample(x=x,
                                     polyphase_logits=logits,
                                     stride=self.stride,
                                     tau=self.gumbel_tau,
                                     mode=mode)
    else:
        # Pass pre-computed probability/logits. Output
        # selected phase and polyphase probability
        logits = None
        out, _prob = self.downsample(x=x,
                                     polyphase_logits=logits,
                                     stride=self.stride,
                                     tau=self.gumbel_tau,
                                     mode=mode,
                                     prob=prob)
    return out, _prob


# Split polyphase components into channels
# TODO: Verify pixel_unshuffle order
def split_polyV1(x, stride, in_channels,
                 num_components):
  _x = nn.functional.pixel_unshuffle(x, stride)
  components = []
  for i in range(num_components):
    components.append(_x[:, i * in_channels: (i + 1) * in_channels, :, :])
  return components


# Polyphase grouped by channels
def split_polyV2_legacy(x, stride, in_channels,
                 num_components):
  _x = nn.functional.pixel_unshuffle(x, stride)
  components = []
  for i in range(num_components):
    components.append(_x[:, i:: num_components, :, :])
  return components

def split_polyV2(x, stride, in_channels,
                 num_components):
  _, _, h, w = x.shape
  x = nn.functional.pad(x, (0, w % stride, 0, h % stride))
  _x = nn.functional.pixel_unshuffle(x, stride)
  bb,_,hh,ww = _x.shape
  return _x.view(bb,in_channels,num_components,hh,ww).permute(2,0,1,3,4)


class PolyphaseInvariantDown2D(nn.Module):
  # TODO: Handle padding for non-divisible shapes
  def __init__(self, stride: int= 2, component_selection=fixed_component,
               pass_extras=False, in_channels=None, hid_channels=None,
               comp_fix_train=False, get_logits=None, antialias_layer=None,
               use_get_logits=True, no_antialias=False, comp_train_convex=False,
               comp_convex=False, logits_pad='circular',selection_noantialias=False):
    super().__init__()
    # comp_fix_train and comp_convex mutually exclusive
    if comp_fix_train: assert not(comp_convex)
    elif comp_convex: assert not(comp_fix_train)

    self.stride = stride
    self.num_components = stride ** 2
    self.pass_extras = pass_extras
    self.comp_fix_train = comp_fix_train
    self.comp_train_convex = comp_train_convex
    self.comp_convex = comp_convex
    self.use_get_logits = use_get_logits
    self.no_antialias = no_antialias
    self.selection_noantialias = selection_noantialias
    if self.no_antialias: self.antialias_layer = None
    else: self.antialias_layer = antialias_layer

    if self.use_get_logits:
      # LPS: No precomputed indices, use get_logits
      self.get_logits = get_logits
      self.logits_pad = logits_pad
    else:
      # LPS: Precomputed indices, no get_logits
      self.get_logits = None
      self.logits_pad = None

    if self.pass_extras:
      # LPS: Pass extra pars
      self.component_selection = component_selection(in_channels=in_channels,
                                                     hid_channels=hid_channels,
                                                     stride=stride,
                                                     get_logits=self.get_logits,
                                                     logits_pad=self.logits_pad)
    else:
      self.component_selection = component_selection

    if self.antialias_layer is not None:
      # Antialiasing layer
      self._antialias_layer = self.antialias_layer(in_channels=in_channels)

  def forward(self, x, ret_prob=False,
              prob=None):
    components_nofilt = None
    if self.antialias_layer is not None:
      if self.selection_noantialias:
        # Keep original components
        in_channels = x.shape[1]
        components_nofilt = split_polyV2(x=x,
                                         stride=self.stride,
                                         in_channels=in_channels,
                                         num_components=self.num_components)

      # Apply antialias
      x = self._antialias_layer(x)

    # Get components
    in_channels = x.shape[1]
    components = split_polyV2(x=x,
                              stride=self.stride,
                              in_channels=in_channels,
                              num_components=self.num_components)

    # Get logits
    if self.pass_extras:
      # LPS: Set gumbel-softmax mode.
      if not(self.training):
        if self.comp_fix_train:
          # Debug: Keep Gumbel-softmax during testing.
          component_mode = "test_gumbel"
        elif self.comp_convex:
          # Softmax
          component_mode = "test_convex"
        else:
          # Argmax
          component_mode = "test"
      else:
        if self.comp_train_convex:
          # Softmax
          component_mode = "train_convex"
        else:
          # Gumbel-softmax (default)
          component_mode = "train"

      # Pass pre-computed prob/logits, components and gumbel-smax mode
      # Pass phases before antialias to compute logits
      # Return optimal phase and polyphase prob/logits
      _x,_p = self.component_selection(x=components,
                                       x_nofilt=components_nofilt,
                                       mode=component_mode,
                                       prob=prob)       
    else:
      # Max_p_norm, fixed_component 
      # Pass pre-computed max index and components
      # Pass phases before antialias to compute indices
      # Return optimal phase and max indices
      _x,_p = self.component_selection(x=components,
                                       x_nofilt=components_nofilt,
                                       prob=prob)

    # Return feats and logits
    if ret_prob: return _x,_p
    else: return _x


class Decimation(nn.Module):
  # Antialias + downsampling
  def __init__(self, stride: int= 2, in_channels=None,
               antialias_layer=None, no_antialias=False):
    super().__init__()
    self.stride = stride
    self.no_antialias = no_antialias
    if self.no_antialias: self.antialias_layer = None
    else: self.antialias_layer = antialias_layer

    if self.antialias_layer is not None:
      # Antialiasing layer
      self._antialias_layer = self.antialias_layer(in_channels=in_channels)

  def forward(self, x):
    if self.antialias_layer is not None:
      # Apply antialias
      x = self._antialias_layer(x)

    # Downsample
    x = x[:,:,::self.stride,::self.stride]
    return x
