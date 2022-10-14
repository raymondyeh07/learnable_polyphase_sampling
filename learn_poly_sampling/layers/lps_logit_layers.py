"""Implements a learnable polyphase down-sampling layer for inferring phase logits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['LPSLogitLayersV2', 'SAInner', 'GraphLogitLayers', 'ComponentPerceptron']

class LPSLogitLayersV1(nn.Module):
  def __init__(self, in_channels, hid_channels, padding_mode):
    super(LPSLogitLayersV1, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, hid_channels, 3,
                           padding='same', padding_mode=padding_mode)
    self.conv2 = nn.Conv2d(hid_channels, hid_channels, 3,
                           padding='same', padding_mode=padding_mode)
    self.relu = nn.ReLU()

  def forward(self, x):
    batch_size = x.shape[0]
    # Get polyphase components
    stride = 2
    xpoly_0 = x[:, :, ::stride, ::stride]
    xpoly_1 = x[:, :, 1::stride, ::stride]
    xpoly_2 = x[:, :, ::stride, 1::stride]
    xpoly_3 = x[:, :, 1::stride, 1::stride]

    # Concatenate to the batch dimension so share weights across.
    xpoly_combined = torch.cat([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim=0)
    ret = self.conv2(self.relu(self.conv1(xpoly_combined)))
    ret = ret.mean(dim=(-1, -2, -3))
    ret = torch.stack(torch.split(ret, batch_size), -1)

    return ret

# Predict polyphase logits
class LPSLogitLayersV2(nn.Module):
  def __init__(self, in_channels, hid_channels,
               padding_mode,skip_connect=False):
    super(LPSLogitLayersV2, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, hid_channels, 3,
                           padding='same', padding_mode=padding_mode)
    self.conv2 = nn.Conv2d(hid_channels, hid_channels, 3,
                           padding='same', padding_mode=padding_mode)
    self.relu = nn.ReLU()
    self.skip_connect = skip_connect

  def forward(self, x):
    batch_size = x[0].shape[0]
    # [BP x C x H x W]
    xpoly_combined = torch.flatten(x,start_dim=0, end_dim=1)
    ret = self.conv2(self.relu(self.conv1(xpoly_combined)))
    if self.skip_connect: ret += xpoly_combined
    ret = ret.mean(dim=(-1, -2, -3))
    ret = torch.stack(torch.split(ret, batch_size), -1)
    return ret

# Predict polyphase logits, include skip connection
class LPSLogitLayersSkip(LPSLogitLayersV2):
  def __init__(self, in_channels, hid_channels,
               padding_mode):
    super().__init__(in_channels=in_channels, hid_channels=hid_channels, padding_mode=padding_mode,
                     skip_connect=True)

# Logits via SA (inner product)
class SAInner(nn.Module):
  def __init__(self, in_channels, hid_channels,
               padding_mode, bn=False):
    super(SAInner, self).__init__()

    # Set batchnorm
    self.bn = bn
    _bn=nn.BatchNorm2d if self.bn else nn.Identity

    # Bias in first conv
    self._bias = not(bn)

    # Layers
    self._phi = nn.Sequential(nn.Conv2d(in_channels, hid_channels, 3, padding=1, padding_mode=padding_mode, bias=self._bias),
                              nn.ReLU(inplace=True),
                              _bn(hid_channels),
                              nn.Conv2d(hid_channels, hid_channels, 3, padding=1, padding_mode=padding_mode))
    self._psi = nn.Sequential(nn.Conv2d(in_channels, hid_channels, 3, padding=1, padding_mode=padding_mode, bias=self._bias),
                              nn.ReLU(inplace=True),
                              _bn(hid_channels),
                              nn.Conv2d(hid_channels, hid_channels, 3, padding=1, padding_mode=padding_mode))
    self._beta = nn.Sequential(nn.Conv2d(in_channels, hid_channels, 3, padding=1, padding_mode=padding_mode, bias=self._bias),
                               nn.ReLU(inplace=True),
                               _bn(hid_channels),
                               nn.Conv2d(hid_channels, hid_channels, 3, padding=1, padding_mode=padding_mode))

  def forward(self, x):
    b = x[0].shape[0]
    p = len(x)
    xpoly_combined = torch.flatten(x,start_dim=0, end_dim=1) #torch.cat(x, dim=0)  # [BP x C x H x W]

    # Affinity
    phi = self._phi(xpoly_combined).mean(
        dim=(-1, -2)).view(p, b, -1).transpose(0, 1)  # [B x P x C]
    psi = self._psi(xpoly_combined).mean(dim=(-1, -2)).view(p,
                                                            b, -1).permute(1, 2, 0)  # [B x C x P]
    inner = torch.bmm(phi, psi)  # [B x P x P]
    inner = F.softmax(inner,dim=-1)

    # Values
    beta = self._beta(xpoly_combined).view(
        p, b, -1).mean(dim=(-1)).transpose(0, 1).unsqueeze(-1)  # [B x P x 1]
    y = torch.bmm(inner, beta).squeeze()
    return y

# Logits via SA (inner product) incl BN
class SAInner_bn(SAInner):
  def __init__(self, in_channels, hid_channels,
               padding_mode):
    super().__init__(in_channels=in_channels, hid_channels=hid_channels, padding_mode=padding_mode,
                     bn=True)

# Predict polyphase logits
class GraphLogitLayers(nn.Module):
  def __init__(self, in_channels, hid_channels,
               padding_mode):
    super(GraphLogitLayers, self).__init__()
    self.phase_size = -1
    self.edge_network = nn.Sequential(nn.Linear(hid_channels*2, hid_channels), nn.ReLU(inplace=True),
                                      nn.Linear(hid_channels, hid_channels), nn.ReLU(inplace=True))
    self.out_network = nn.Sequential(nn.Linear(hid_channels, hid_channels),
                                     nn.ReLU(inplace=True), nn.Linear(hid_channels, 1))
    self.input_conv = nn.Sequential(nn.Conv2d(in_channels, hid_channels, 3,
                                    padding='same', padding_mode=padding_mode), nn.ReLU(inplace=True))


  def setup_send_receive_matrix(self, phase_size, device):
    if phase_size != self.phase_size:
      adj_mat = torch.ones(phase_size, phase_size, device=device)
      self.send_mat = torch.nn.functional.one_hot(torch.where(adj_mat)[0]).float()
      self.receive_mat = torch.nn.functional.one_hot(torch.where(adj_mat)[1]).float()
      self.phase_size = phase_size

  def node2edge(self, x):
    send = torch.matmul(self.send_mat, x)
    receive = torch.matmul(self.receive_mat, x)
    edges = torch.cat([send, receive], dim=-1)
    return edges

  def edge2node(self,x):
    msg = torch.matmul(self.receive_mat.t(), x)
    return msg / (1.*self.receive_mat.size(-1))

  def forward(self, x):
    batch_size = x[0].shape[0]
    phase_size = len(x)
    xpoly_combined = torch.flatten(x,start_dim=0, end_dim=1)
    xpoly_combined = self.input_conv(xpoly_combined).mean(
        dim=(-1, -2)).view(phase_size, batch_size, -1).transpose(0,1)
    self.setup_send_receive_matrix(phase_size, device=xpoly_combined.device)
    # Graph layers.
    edges = self.node2edge(xpoly_combined)
    edges = self.edge_network(edges)
    nodes = self.edge2node(edges)  # [B x H x W x P x C]
    # Output
    ret = self.out_network(nodes).squeeze()
    return ret

class ComponentPerceptron(nn.Module):
  def __init__(self, in_channels, reduce=True, **kwargs):
    super().__init__()
    self.perceptron = nn.Conv2d(in_channels, 1, 1)
    self.reduce = reduce

  def forward(self, x):
    batch_size = x[0].shape[0]
    xpoly_combined = torch.flatten(x,start_dim=0, end_dim=1)
    ret = self.perceptron(xpoly_combined)
    if self.reduce:
      ret = ret.mean(dim=(-1, -2, -3))
    ret = torch.stack(torch.split(ret, batch_size), 1)
    return ret
