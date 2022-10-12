import torch
import unittest
from layers import PolyphaseInvariantDown2D,LPS,get_antialias
from layers.polydown import split_polyV2,max_p_norm
from layers.polyup import PolyphaseInvariantUp2D,LPS_u,max_p_norm_u
from layers.lps_utils import lps_downsampleV2, lps_upsampleV2
from layers.lps_logit_layers import LPSLogitLayersV2


class TestShiftEquivariance(unittest.TestCase):
  # (Core) LPS pool - unpool
  def test_LPS_pool_unpool(self):
    # LPS: test mode (argmax)
    mode = 'test'
    hard = True
    tau = 1

    # Input
    stride = 2
    b,c,h,w = 4,3,512,512
    num_components = stride**2
    x = torch.arange(b*c*h*w).reshape(b,c,h,w).float()+1
    x_shift = torch.roll(x,shifts=(-1,-1),dims=(2,3))

    # Polyphase decomposition
    comps = split_polyV2(x=x,
                         stride=stride,
                         in_channels=c,
                         num_components=num_components)
    comps_shift = split_polyV2(x=x_shift,
                               stride=stride,
                               in_channels=c,
                               num_components=num_components)

    # Logits
    logits = torch.randn(b,4)
    prob = torch.nn.functional.softmax(logits,dim=-1)
    logits_shift= logits[:,[3,2,1,0]]
    prob_shift = torch.nn.functional.softmax(logits_shift,dim=-1)

    # Pool
    out1,_ = lps_downsampleV2(x=comps,
                              stride=stride,
                              polyphase_logits=logits,
                              mode=mode,
                              hard=hard,
                              tau=tau)
    out1_shift,_ = lps_downsampleV2(x=comps_shift,
                                    stride=stride,
                                    polyphase_logits=logits_shift,
                                    mode=mode,
                                    hard=hard,
                                    tau=tau)

    # Unpool
    x_up,_ = lps_upsampleV2(x=out1,
                            stride=stride,
                            prob=prob,
                            mode=mode)
    x_shift_up,_ = lps_upsampleV2(x=out1_shift,
                                  stride=stride,
                                  prob=prob_shift,
                                  mode=mode)

    # Check equivariance
    x_unshift_up = torch.roll(x_shift_up,shifts=(1,1),dims=(2,3))
    assert torch.allclose(x_up,x_unshift_up)

  # (Core) LPS pool - unpool, random input
  def test_LPS_pool_unpool_random(self):
    # LPS: test mode (argmax)
    mode = 'test'
    hard = True
    tau = 1

    # Input
    stride = 2
    b,c,h,w = 4,3,512,512
    num_components = stride**2
    x = torch.randn(b,c,h,w).float()
    x_shift = torch.roll(x,shifts=(-1,-1),dims=(2,3))

    # Polyphase decomposition
    comps = split_polyV2(x=x,
                         stride=stride,
                         in_channels=c,
                         num_components=num_components)
    comps_shift = split_polyV2(x=x_shift,
                               stride=stride,
                               in_channels=c,
                               num_components=num_components)

    # Logits
    model = LPSLogitLayersV2(in_channels=c,
                             hid_channels=c,
                             padding_mode='circular')
    logits = model(comps)
    prob = torch.nn.functional.softmax(logits,dim=-1)
    logits_shift = model(comps_shift)
    prob_shift = torch.nn.functional.softmax(logits_shift,dim=-1)

    # Pool
    out1,_ = lps_downsampleV2(x=comps,
                              stride=stride,
                              polyphase_logits=logits,
                              mode=mode,
                              hard=hard,
                              tau=tau)
    out1_shift,_ = lps_downsampleV2(x=comps_shift,
                                    stride=stride,
                                    polyphase_logits=logits_shift,
                                    mode=mode,
                                    hard=hard,
                                    tau=tau)

    # Unpool
    x_up,_ = lps_upsampleV2(x=out1,
                            stride=stride,
                            prob=prob,
                            mode='test')
    x_shift_up,_ = lps_upsampleV2(x=out1_shift,
                                  stride=stride,
                                  prob=prob_shift,
                                  mode='test')

    # Check equivariance
    x_unshift_up = torch.roll(x_shift_up,shifts=(1,1),dims=(2,3))
    assert torch.allclose(x_up,x_unshift_up)

  # Downsampling - upsampling
  def test_poly_down_up(self):
    # Input
    stride = 2
    b,c,h,w = 4,3,512,512
    x = torch.randn(b,c,h,w).float()
    x_shift = torch.roll(x,shifts=(-1,-1),dims=(2,3))

    # LPS
    down = PolyphaseInvariantDown2D(stride=stride,
                                    component_selection=LPS,
                                    pass_extras=True,
                                    in_channels=c,
                                    hid_channels=c,
                                    get_logits=LPSLogitLayersV2)
    up = PolyphaseInvariantUp2D(stride=stride,
                                component_selection=LPS_u,
                                pass_extras=True,
                                in_channels=c,
                                hid_channels=c)
    down.eval()
    up.eval()

    # Pool
    compsV2,p = down(x=x,ret_prob=True)
    compsV2_shift,p_shift = down(x=x_shift,ret_prob=True)

    # Unpool
    x_r = up(x=compsV2,prob=p)
    x_shift_r = up(x=compsV2_shift,prob=p_shift)

    # Check equivariance
    x_unshift_r = torch.roll(x_shift_r,shifts=(1,1),dims=(2,3))
    assert torch.allclose(x_r,x_unshift_r)

  # Decimation - interpolation
  def test_poly_dec_int(self):
    # Antialias pars
    antialias_mode = 'LowPassFilter'
    antialias_size = 5
    antialias_padding = 'same'
    antialias_padding_mode = 'circular'
    antialias_group = None

    # Input
    stride = 2
    b,c,h,w = 4,3,512,512
    x = torch.randn(b,c,h,w).float()
    x_shift = torch.roll(x,shifts=(-1,-1),dims=(2,3))

    # Antialias filter
    antialias = get_antialias(antialias_mode=antialias_mode,
                              antialias_size=antialias_size,
                              antialias_padding=antialias_padding,
                              antialias_padding_mode=antialias_padding_mode,
                              antialias_group=antialias_group)

    # LPS
    down = PolyphaseInvariantDown2D(stride=stride,
                                    component_selection=LPS,
                                    pass_extras=True,
                                    in_channels=c,
                                    hid_channels=c,
                                    get_logits=LPSLogitLayersV2,
                                    antialias_layer=antialias)
    up = PolyphaseInvariantUp2D(stride=stride,
                                component_selection=LPS_u,
                                pass_extras=True,
                                in_channels=c,
                                hid_channels=c,
                                antialias_layer=antialias)
    down.eval()
    up.eval()

    # Pool
    comps,p = down(x=x,ret_prob=True)
    comps_shift,p_shift = down(x=x_shift,ret_prob=True)

    # Unpool
    x_r = up(x=comps,prob=p)
    x_shift_r = up(x=comps_shift,prob=p_shift)

    # Check equivariance
    x_unshift_r = torch.roll(x_shift_r,shifts=(1,1),dims=(2,3))
    assert torch.allclose(x_r,x_unshift_r)

  # (Core) M2N pool - unpool, random input
  def test_M2N_pool_unpool_random(self):
    # Input
    stride = 2
    b,c,h,w = 4,3,512,512
    num_components = stride**2
    x = torch.randn(b,c,h,w).float()
    x_shift = torch.roll(x,shifts=(-1,-1),dims=(2,3))

    # Polyphase decomposition
    comps = split_polyV2(x=x,
                         stride=stride,
                         in_channels=c,
                         num_components=num_components)
    comps_shift = split_polyV2(x=x_shift,
                               stride=stride,
                               in_channels=c,
                               num_components=num_components)

    # Pool
    out1,idx = max_p_norm(x=comps)
    out1_shift,idx_shift = max_p_norm(x=comps_shift)

    # Unpool
    x_up,_ = max_p_norm_u(x=out1,
                          stride=stride,
                          prob=idx)
    x_shift_up,_ = max_p_norm_u(x=out1_shift,
                                stride=stride,
                                prob=idx_shift)

    # Check equivariance
    x_unshift_up = torch.roll(x_shift_up,shifts=(1,1),dims=(2,3))
    assert torch.allclose(x_up,x_unshift_up)

if __name__ == '__main__':
  unittest.main()
