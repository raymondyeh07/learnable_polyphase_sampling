import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet_custom import cpad
import unittest


class TestDilatedConv(unittest.TestCase):
  # Check dilated convolution shift invariance
  def check_dilated(self,x,s,
                    dilation,padding_mode,kernel_size=3,
                    bias=False,custom_padding=False):
    c = x.shape[1]
    x_shift = torch.roll(x,shifts=(s[0],s[1]),dims=(2,3))

    if custom_padding:
      assert padding_mode in ['zeros','circular']
      if padding_mode=='zeros':
        pad = nn.ZeroPad2d((dilation))
      else:
        pad = cpad(pad=[dilation,dilation,dilation,dilation])
      _C = nn.Conv2d(in_channels=c,
                     out_channels=c,
                     kernel_size=kernel_size,
                     dilation=dilation,
                     bias=bias)
      C = nn.Sequential(pad,_C)
    else:
      C = nn.Conv2d(in_channels=c,
                    out_channels=c,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=dilation,
                    padding_mode=padding_mode,
                    bias=bias)

    # Output
    y = C(x)
    y_shift = C(x_shift)
    y_ref = torch.roll(y,shifts=(s[0],s[1]),dims=(2,3))
    errval = torch.norm(y_shift-y_ref)
    print("[check_dilated] errval: ",errval)
    assert torch.allclose(y_shift,y_ref)

  def test_N7_d6(self):
    # Check ASPPConv layer, in_size=7, dilation=6
    s = [-5,-5]
    b,c,h,w = 4,16,7,7
    dilation = 6
    padding_mode = 'circular'
    x = torch.randn(b,c,h,w)
    self.check_dilated(x=x,
                       s=s,
                       dilation=dilation,
                       padding_mode=padding_mode,
                       )
    return

  def test_N7_d12(self):
    # Check ASPPConv layer, in_size=7, dilation=6
    s = [-5,-5]
    b,c,h,w = 4,16,7,7
    dilation = 12
    padding_mode = 'zeros'
    x = torch.randn(b,c,h,w)
    self.check_dilated(x=x,
                       s=s,
                       dilation=dilation,
                       padding_mode=padding_mode,
                       custom_padding=True,
                       )
    return

  def test_N7_d18(self):
    # Check ASPPConv layer, in_size=7, dilation=6
    s = [-5,-5]
    b,c,h,w = 4,16,7,7
    dilation = 18
    padding_mode = 'zeros'
    x = torch.randn(b,c,h,w)
    self.check_dilated(x=x,
                       s=s,
                       dilation=dilation,
                       padding_mode=padding_mode,
                       custom_padding=True,
                       )
    return

if __name__ == '__main__':
  unittest.main()
