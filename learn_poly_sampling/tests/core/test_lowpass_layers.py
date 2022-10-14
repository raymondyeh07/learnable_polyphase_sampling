import unittest
import torch
import torch.nn as nn
import numpy as np
from layers import LowPassFilter


# Seed for reproducibility
import random
import numpy as np

random.seed(0)
np.random.seed(0)
FILTER_SIZES = range(1, 8)
torch.manual_seed(0)

#Bed of nails upsampling
class bed_of_nails( nn.Module):
  def __init__( self):
    super( bed_of_nails, self).__init__()

  def forward( self, x, stride):
    [ b, c, w, h]= x.shape
    out= torch.zeros( b, c, stride* w, stride* h)
    out[ :, :, ::stride, ::stride]= x
    return out

class TestLowPassFilter(unittest.TestCase):

    def setUp(self):
        # K. size 2
        xx = np.array([[1, 0, 1, 0, 2, 0],
                       [0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 2, 0],
                       [0, 0, 0, 0, 0, 0],
                       ],
                       dtype='float32')
        xx = np.stack([xx, xx])
        xx = xx[np.newaxis, :]
        self.xx = torch.Tensor(xx)

        # All Kernels
        x=np.array([[1, 1, 2],
                    [1, 1, 2],
                    ],
                    dtype='float32')
        x=np.stack([x,x])
        x=x[np.newaxis,:]
        self.x=torch.Tensor(x)
        self.upsample=bed_of_nails()

        # Target output for larger k
        self.y_list=[]
        self.y_list.append(np.array([[1., 1., 2.],
                                     [1., 1., 2.],
                                    ],
                                    dtype='float32'))

        self.y_list.append(np.array([[.25, .25, .25, .5, .5, .25],
                                     [.25, .25, .25, .5, .5, .25],
                                     [.25, .25, .25, .5, .5, .25],
                                     [.25, .25, .25, .5, .5, .25],
                                    ],
                                    dtype='float32'))

        self.y_list.append(np.array([[.25, .125, .125, .25, .125, .25, .5000, .25, .125],
                                     [.125, .0625, .0625, .125, .0625, .125, .25, .125, .0625],
                                     [.125, .0625, .0625, .125, .0625, .125, .25, .125, .0625],
                                     [.25, .125, .125, .25, .125, .25, .5000, .25, .125],
                                     [.125, .0625, .0625, .125, .0625, .125, .25, .125, .0625],
                                     [.125, .0625, .0625, .125, .0625, .125, .25, .125, .0625],
                                    ],
                                    dtype='float32'))

        self.y_list.append(np.array([[.140625, .046875, .046875, .140625, .140625, .046875, .09375, .28125, .28125, .09375, .046875, .140625],
                                     [.046875, .015625, .015625, .046875, .046875, .015625, .03125, .09375, .09375, .03125, .015625, .046875],
                                     [.046875, .015625, .015625, .046875, .046875, .015625, .03125, .09375, .09375, .03125, .015625, .046875],
                                     [.140625, .046875, .046875, .140625, .140625, .046875, .09375, .28125, .28125, .09375, .046875, .140625],
                                     [.140625, .046875, .046875, .140625, .140625, .046875, .09375, .28125, .28125, .09375, .046875, .140625],
                                     [.046875, .015625, .015625, .046875, .046875, .015625, .03125, .09375, .09375, .03125, .015625, .046875],
                                     [.046875, .015625, .015625, .046875, .046875, .015625, .03125, .09375, .09375, .03125, .015625, .046875],
                                     [.140625, .046875, .046875, .140625, .140625, .046875, .09375, .28125, .28125, .09375, .046875, .140625],
                                    ],
                                    dtype='float32'))

        self.y_list.append(np.array([[.140625, .09375, .0234375, .0234375, .09375, .140625, .09375, .0234375, .046875, .1875, .28125, .1875, .046875, .0234375, .09375],
                                     [.09375, .0625, .015625, .015625, .0625, .09375, .0625, .015625, .03125, .1250, .1875, .1250, .03125, .015625, .0625],
                                     [.0234375, .015625, .00390625, .00390625, .015625, .0234375, .015625, .00390625, .0078125, .03125, .046875, .03125, .0078125, .00390625, .015625],
                                     [.0234375, .015625, .00390625, .00390625, .015625, .0234375, .015625, .00390625, .0078125, .03125, .046875, .03125, .0078125, .00390625, .015625],
                                     [.09375, .0625, .015625, .015625, .0625, .09375, .0625, .015625, .03125, .1250, .1875, .1250, .03125, .015625, .0625],
                                     [.140625, .09375, .0234375, .0234375, .09375, .140625, .09375, .0234375, .046875, .1875, .28125, .1875, .046875, .0234375, .09375],
                                     [.09375, .0625, .015625, .015625, .0625, .09375, .0625, .015625, .03125, .1250, .1875, .1250, .03125, .015625, .0625],
                                     [.0234375, .015625, .00390625, .00390625, .015625, .0234375, .015625, .00390625, .0078125, .03125, .046875, .03125, .0078125, .00390625, .015625],
                                     [.0234375, .015625, .00390625, .00390625, .015625, .0234375, .015625, .00390625, .0078125, .03125, .046875, .03125, .0078125, .00390625, .015625],
                                     [.09375, .0625, .015625, .015625, .0625, .09375, .0625, .015625, .03125, .1250, .1875, .1250, .03125, .015625, .0625],
                                    ],
                                    dtype='float32'))

        self.y_list.append(np.array([[0.09765625, 0.04882812, 0.00976562, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.01953125, 0.09765625, 0.1953125, 0.1953125, 0.09765625, 0.01953125, 0.00976562, 0.04882812, 0.09765625],
                                     [0.04882812, 0.02441406, 0.00488281, 0.00488281, 0.02441406, 0.04882812, 0.04882812, 0.02441406, 0.00488281, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.00488281, 0.02441406, 0.04882812],
                                     [0.00976562, 0.00488281, 0.00097656, 0.00097656, 0.00488281, 0.00976562, 0.00976562, 0.00488281, 0.00097656, 0.00195312, 0.00976562, 0.01953125, 0.01953125, 0.00976562, 0.00195312, 0.00097656, 0.00488281, 0.00976562],
                                     [0.00976562, 0.00488281, 0.00097656, 0.00097656, 0.00488281, 0.00976562, 0.00976562, 0.00488281, 0.00097656, 0.00195312, 0.00976562, 0.01953125, 0.01953125, 0.00976562, 0.00195312, 0.00097656, 0.00488281, 0.00976562],
                                     [0.04882812, 0.02441406, 0.00488281, 0.00488281, 0.02441406, 0.04882812, 0.04882812, 0.02441406, 0.00488281, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.00488281, 0.02441406, 0.04882812],
                                     [0.09765625, 0.04882812, 0.00976562, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.01953125, 0.09765625, 0.1953125, 0.1953125, 0.09765625, 0.01953125, 0.00976562, 0.04882812, 0.09765625],
                                     [0.09765625, 0.04882812, 0.00976562, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.01953125, 0.09765625, 0.1953125, 0.1953125, 0.09765625, 0.01953125, 0.00976562, 0.04882812, 0.09765625],
                                     [0.04882812, 0.02441406, 0.00488281, 0.00488281, 0.02441406, 0.04882812, 0.04882812, 0.02441406, 0.00488281, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.00488281, 0.02441406, 0.04882812],
                                     [0.00976562, 0.00488281, 0.00097656, 0.00097656, 0.00488281, 0.00976562, 0.00976562, 0.00488281, 0.00097656, 0.00195312, 0.00976562, 0.01953125, 0.01953125, 0.00976562, 0.00195312, 0.00097656, 0.00488281, 0.00976562],
                                     [0.00976562, 0.00488281, 0.00097656, 0.00097656, 0.00488281, 0.00976562, 0.00976562, 0.00488281, 0.00097656, 0.00195312, 0.00976562, 0.01953125, 0.01953125, 0.00976562, 0.00195312, 0.00097656, 0.00488281, 0.00976562],
                                     [0.04882812, 0.02441406, 0.00488281, 0.00488281, 0.02441406, 0.04882812, 0.04882812, 0.02441406, 0.00488281, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.00488281, 0.02441406, 0.04882812],
                                     [0.09765625, 0.04882812, 0.00976562, 0.00976562, 0.04882812, 0.09765625, 0.09765625, 0.04882812, 0.00976562, 0.01953125, 0.09765625, 0.1953125, 0.1953125, 0.09765625, 0.01953125, 0.00976562, 0.04882812, 0.09765625],
                                    ],
                                    dtype='float32'))

        self.y_list.append(np.array([[0.09765625, 0.07324219, 0.02929688, 0.00488281, 0.00488281, 0.02929688, 0.07324219, 0.09765625, 0.07324219, 0.02929688, 0.00488281, 0.00976562, 0.05859375, 0.14648438, 0.1953125, 0.14648438, 0.05859375, 0.00976562, 0.00488281, 0.02929688, 0.07324219],
                                     [0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00366211, 0.02197266, 0.05493164, 0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00732422, 0.04394531, 0.10986328, 0.14648438, 0.10986328, 0.04394531, 0.00732422, 0.00366211, 0.02197266, 0.05493164],
                                     [0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00146484, 0.00878906, 0.02197266, 0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00292969, 0.01757812, 0.04394531, 0.05859375, 0.04394531, 0.01757812, 0.00292969, 0.00146484, 0.00878906, 0.02197266],
                                     [0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00024414, 0.00146484, 0.00366211, 0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00048828, 0.00292969, 0.00732422, 0.00976562, 0.00732422, 0.00292969, 0.00048828, 0.00024414, 0.00146484, 0.00366211],
                                     [0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00024414, 0.00146484, 0.00366211, 0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00048828, 0.00292969, 0.00732422, 0.00976562, 0.00732422, 0.00292969, 0.00048828, 0.00024414, 0.00146484, 0.00366211],
                                     [0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00146484, 0.00878906, 0.02197266, 0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00292969, 0.01757812, 0.04394531, 0.05859375, 0.04394531, 0.01757812, 0.00292969, 0.00146484, 0.00878906, 0.02197266],
                                     [0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00366211, 0.02197266, 0.05493164, 0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00732422, 0.04394531, 0.10986328, 0.14648438, 0.10986328, 0.04394531, 0.00732422, 0.00366211, 0.02197266, 0.05493164],
                                     [0.09765625, 0.07324219, 0.02929688, 0.00488281, 0.00488281, 0.02929688, 0.07324219, 0.09765625, 0.07324219, 0.02929688, 0.00488281, 0.00976562, 0.05859375, 0.14648438, 0.1953125, 0.14648438, 0.05859375, 0.00976562, 0.00488281, 0.02929688, 0.07324219],
                                     [0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00366211, 0.02197266, 0.05493164, 0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00732422, 0.04394531, 0.10986328, 0.14648438, 0.10986328, 0.04394531, 0.00732422, 0.00366211, 0.02197266, 0.05493164],
                                     [0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00146484, 0.00878906, 0.02197266, 0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00292969, 0.01757812, 0.04394531, 0.05859375, 0.04394531, 0.01757812, 0.00292969, 0.00146484, 0.00878906, 0.02197266],
                                     [0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00024414, 0.00146484, 0.00366211, 0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00048828, 0.00292969, 0.00732422, 0.00976562, 0.00732422, 0.00292969, 0.00048828, 0.00024414, 0.00146484, 0.00366211],
                                     [0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00024414, 0.00146484, 0.00366211, 0.00488281, 0.00366211, 0.00146484, 0.00024414, 0.00048828, 0.00292969, 0.00732422, 0.00976562, 0.00732422, 0.00292969, 0.00048828, 0.00024414, 0.00146484, 0.00366211],
                                     [0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00146484, 0.00878906, 0.02197266, 0.02929688, 0.02197266, 0.00878906, 0.00146484, 0.00292969, 0.01757812, 0.04394531, 0.05859375, 0.04394531, 0.01757812, 0.00292969, 0.00146484, 0.00878906, 0.02197266],
                                     [0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00366211, 0.02197266, 0.05493164, 0.07324219, 0.05493164, 0.02197266, 0.00366211, 0.00732422, 0.04394531, 0.10986328, 0.14648438, 0.10986328, 0.04394531, 0.00732422, 0.00366211, 0.02197266, 0.05493164],
                                    ],
                                    dtype='float32'))

    def test_output_sizes(self):
      for sz in FILTER_SIZES:
        lowpass = LowPassFilter(in_channels=2,
                                filter_size=sz,
                                padding='same',
                                padding_mode='reflect')
        yy = lowpass(self.xx)
        assert yy.shape == self.xx.shape

    def test_2x2_lowpass(self):
      lowpass = LowPassFilter(in_channels=2,
                              filter_size=2,
                              padding='same',
                              padding_mode='circular')

      yy = lowpass(self.xx)
      _yy = np.array([[.25, .25, .25, .5, .5, .25],
                      [.25, .25, .25, .5, .5, .25],
                      [.25, .25, .25, .5, .5, .25],
                      [.25, .25, .25, .5, .5, .25],
                     ],
                     dtype='float32')
      _yy = np.stack([_yy, _yy])
      _yy = _yy[np.newaxis, :]

    def test_nxn_lowpass(self):
      for i in range( len( FILTER_SIZES)):
        # Set lowpass
        k=FILTER_SIZES[i]
        lowpass=LowPassFilter(in_channels=2,
                              filter_size=k,
                              padding='same',
                              padding_mode='circular')

        # Upsample input
        x=self.upsample(x=self.x,
                        stride=k)

        # Lowpass output
        y=lowpass(x).detach().cpu().numpy()

        # Target
        _y=self.y_list[i]
        _y=np.stack([_y,_y])
        _y=_y[np.newaxis,:]
        assert np.allclose(y,_y)

    def test_constant_weights(self):

      class DummyModule(nn.Module):
        def __init__(self):
          super().__init__()
          self.lowpass = LowPassFilter(in_channels=2,
                                       filter_size=2,
                                       padding='same',
                                       padding_mode='circular')
          self.flatten = nn.Flatten()
          self.dense = nn.Linear(48, 1)

        def forward(self, x):
          x = self.lowpass(x)
          x = self.flatten(x)
          x = self.dense(x)
          return x

      dummyModule = DummyModule()

      assert dummyModule.lowpass.filt.requires_grad == False

      initial_filter = dummyModule.lowpass.filt.detach().clone().numpy()
      initial_dense_w = dummyModule.dense.weight[0].detach().clone().numpy()
      initial_dense_b = dummyModule.dense.bias[0].detach().clone().numpy()

      opt = torch.optim.SGD(dummyModule.parameters(), 0.1)

      opt.zero_grad()
      y = dummyModule(self.xx)
      loss = (y + 1)**2
      loss.backward()
      opt.step()


      assert np.allclose(initial_filter, dummyModule.lowpass.filt.detach().numpy())
      assert not np.allclose(initial_dense_w, dummyModule.dense.weight[0].detach().numpy())
      assert not np.allclose(initial_dense_b, dummyModule.dense.bias[0].detach().numpy())
