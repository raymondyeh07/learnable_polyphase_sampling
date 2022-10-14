import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CircularPad2d(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad_v = 4*[pad]
    def extra_repr(self):
        return ("pad_v={pad_v}".format(pad_v=self.pad_v))
    def forward(self, inp):
        return F.pad(inp, pad=self.pad_v, mode='circular')

class LowPassFilter(nn.Module):
    def __init__(self, in_channels, filter_size,
                 padding, padding_mode, filter_scale=1):
        super().__init__()
        self.filter_size = filter_size
        self.padding_mode = padding_mode
        self.padding = padding
        self.channels = in_channels

        if(self.filter_size == 1):
            a = np.array([[	1.,]])
        elif(self.filter_size==2):
            a = np.array([[	1., 1.]])
        elif(self.filter_size==3):
            a = np.array([[	1., 2., 1.]])
        elif(self.filter_size==4):
            a = np.array([[	1., 3., 3., 1.]])
        elif(self.filter_size==5):
            a = np.array([[	1., 4., 6., 4., 1.]])
        elif(self.filter_size==6):
            a = np.array([[	1., 5., 10., 10., 5., 1.]])
        elif(self.filter_size==7):
            a = np.array([[	1., 6., 15., 20., 15., 6., 1.]])
        else:
            raise ValueError('Filter size must be 1-7', self.filter_size)

        filt = a * a.T
        filt = torch.Tensor(filt/np.sum(filt))
        filt *= (filter_scale**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat(self.channels, 1, 1, 1))

        _allowed_padding = ('valid', 'same')
        if self.padding == 'valid':
           self.pad_tuple = None
        elif self.padding == 'same':
            _pad = (self.filter_size - 1)/2
            _pad_l = int(np.floor(_pad))
            _pad_r = int(np.ceil(_pad))
            self.pad_tuple = (_pad_l, _pad_r, _pad_l, _pad_r)
        else:
            raise ValueError(f'padding must be one of {_allowed_padding}', self.padding)

    def extra_repr(self):
        return ("in_channels={in_channels}, filter_size={filter_size}, padding={padding}, "
                "padding_mode={padding_mode}".format(in_channels=self.channels,
                filter_size=self.filter_size, padding=self.padding, padding_mode=self.padding_mode))

    def forward(self, inp):
        if self.padding != 'valid':
            inp = F.pad(inp, self.pad_tuple, self.padding_mode)
        return F.conv2d(inp, self.filt, groups=inp.shape[1])

# Adaptive Antialiasing
# Ref: https://github.com/MaureenZOU/Adaptive-anti-Aliasing
def get_pad_layer(pad_type):
    if pad_type=='reflect':
        PadLayer = nn.ReflectionPad2d
    elif pad_type=='replicate':
        PadLayer = nn.ReplicationPad2d
    elif pad_type=='zero':
        PadLayer = nn.ZeroPad2d
    elif pad_type=='circular':
        PadLayer = CircularPad2d
    else:
        raise ValueError("Pad type %s not recognized"%pad_type)
    return PadLayer

class DDAC(nn.Module):
    def __init__(self, in_channels, kernel_size,
                 pad_type='reflect', group=2, kernel_scale=1):
        super(DDAC, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.kernel_size = kernel_size
        self.kernel_scale = kernel_scale
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x*(self.kernel_scale**2)
