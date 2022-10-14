"""Implements learnable polyphase down-sampling layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lps_downsample(x, stride, polyphase_logits, mode='train', tau=1.0, hard=False):
  # Only supports stride of 2
  assert stride == 2

  # TODO: Can also be per-channel.
  xpoly_0 = x[:, :, ::stride, ::stride]
  xpoly_1 = x[:, :, 1::stride, ::stride]
  xpoly_2 = x[:, :, ::stride, 1::stride]
  xpoly_3 = x[:, :, 1::stride, 1::stride]
  xpoly_combined = torch.stack([xpoly_0, xpoly_1, xpoly_2, xpoly_3], dim=4)
  batch_size, phase_size = polyphase_logits.shape
  if mode == 'train':
    sampled_prob = F.gumbel_softmax(polyphase_logits, tau=tau, hard=hard)
    sampled_prob = sampled_prob.view(batch_size, 1, 1, 1, phase_size)
    ret = torch.sum(xpoly_combined*sampled_prob, -1)
  else:
    _, sampled_idx = torch.max(polyphase_logits, -1)
    ret = xpoly_combined[torch.arange(batch_size), :, :, :, sampled_idx]
  return ret


def lps_downsampleV2( x, stride, polyphase_logits,
                      mode= 'train', tau= 1.0, hard= False,
                      prob=None):
    # Only supports stride of 2
    assert stride==2
    assert mode in ['train','train_convex','test','test_convex','test_gumbel']

    # Convert to tensor
    xpoly_combined= x.permute(1,2,3,4,0) #torch.stack( x, dim= 4)
    batch_size, phase_size= x[0].shape[0], len(x)
    if mode in ["train","test_gumbel"]:
        if polyphase_logits is None:
            # Use precomputed probability
            # Prob holds polyphase probabilities
            _sampled_prob= prob
        else:
            # Compute probability
            _sampled_prob= F.gumbel_softmax( polyphase_logits,
                                             tau= tau,
                                             hard= hard)
        sampled_prob= _sampled_prob.view( batch_size, 1, 1, 1, phase_size)
        ret= torch.sum( xpoly_combined* sampled_prob, -1)
    elif mode=="test":
        if polyphase_logits is None:
            # Keep max phase, use precomputed logits
            # Prob holds polyphase logits
            polyphase_logits=prob
        # Keep max phase
        _, sampled_idx= torch.max( polyphase_logits, -1)
        ret= xpoly_combined[ torch.arange( batch_size), :, :, :, sampled_idx]
    elif mode in ["train_convex","test_convex"]:
        if polyphase_logits is None:
            # Use precomputed probability
            # Prob holds polyphase probabilities
            _sampled_prob= prob
        else:
            # Compute weights
            _sampled_prob= F.softmax( polyphase_logits/tau,
                                      dim=-1)
        sampled_prob= _sampled_prob.view( batch_size, 1, 1, 1, phase_size)
        ret= torch.sum( xpoly_combined* sampled_prob, -1)

    # Output phase
    if mode in ["train","train_convex"]:
        # Output phase and prob\logits tuple
        # Notice logits=None if pre-computed prob passed
        return ret,(_sampled_prob,polyphase_logits)
    elif mode in ["test_convex","test_gumbel"]:
        # Output phase and prob
        return ret, _sampled_prob
    elif mode== "test":
        # Output also polyphase logits
        return ret, polyphase_logits


def lps_upsample(x, stride, polyphase_logits=None,
                 mode='train', tau=1.0, hard=False,
                 prob=None):
  """LPS Upsampling Layer.
  Args:
      x ([tensor]): Batch Size x Channel x Height x Width.
      polyphase_logits: Batch size x number of phases.
  """
  assert stride == 2  # TODO: Implement support for different stride other than 2.
  if mode == 'train':
    bb,cc,hh,ww = x.shape
    x_poly = torch.repeat_interleave(x,stride**2,dim=1)
    prob_poly = prob.repeat([1,cc]).view(bb,cc*stride**2,1,1)
    # Multiply the logits here.
    x_poly_weighted = x_poly*prob_poly
    x_up = nn.functional.pixel_shuffle(x_poly_weighted, stride)
  elif mode == 'test':
    bb,cc,hh,ww = x.shape
    x_poly = torch.repeat_interleave(x,stride**2,dim=1)
    argmax_idx = torch.argmax(polyphase_logits,-1)
    hard_prob = torch.nn.functional.one_hot(argmax_idx,stride**2)
    hard_prob_poly = hard_prob.repeat([1,cc]).view(bb,cc*stride**2,1,1)
    x_poly_weighted = x_poly*hard_prob_poly
    x_up = nn.functional.pixel_shuffle(x_poly_weighted, stride)
  else:
    raise ValueError("Undefined mode. Check 'mode' input argument.")
  return x_up


def lps_upsampleV2(x,prob,stride,
                   mode='train',get_samples=False,tau=1.0,
                   hard=False):
  """ LPS Upsampling Layer
  Args:
      1. x ([tensor]): Batch Size x Channel x Height x Width.
      2. prob: if mode='train', tuple. prob[0]: gumbel-softmax probability from pooling layer,
                                       prob[1]: polyphase logits from pooling layer.
               else if mode='test', polyphase logits from pooling layer.     
  """
  assert stride == 2  # TODO: Implement support for different stride factors.
  assert mode in ['train','test','test_convex']

  bb,cc,hh,ww = x.shape
  x_poly = torch.repeat_interleave(x,stride**2,dim=1)
  if mode == 'train':
    if get_samples:
      # Sample probability via gumbel-softmax
      sampled_prob = F.gumbel_softmax(prob[1],
                                      tau=tau,
                                      hard=hard)
    else:
      # Use pre-computed probability
      sampled_prob = prob[0]
    prob_poly = sampled_prob.repeat([1,cc]).view(bb,cc*stride**2,1,1)
    x_poly_weighted = x_poly*prob_poly
  elif mode == 'test':
    # Hard selection. prob = polyphase logits
    argmax_idx = torch.argmax(prob,-1)
    hard_prob = torch.nn.functional.one_hot(argmax_idx,stride**2)
    hard_prob_poly = hard_prob.repeat([1,cc]).view(bb,cc*stride**2,1,1)
    x_poly_weighted = x_poly*hard_prob_poly
  elif mode == 'test_convex':
    # Soft selection, use pre-computed logits
    sampled_prob = F.softmax(prob/tau,
                             dim=-1)
    prob_poly = sampled_prob.repeat([1,cc]).view(bb,cc*stride**2,1,1)
    x_poly_weighted = x_poly*prob_poly

  x_up = nn.functional.pixel_shuffle(x_poly_weighted, stride)
  return x_up,prob

