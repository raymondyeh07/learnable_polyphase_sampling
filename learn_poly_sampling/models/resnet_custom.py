import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from .core import AbstractBaseClassifierModel
from layers.polydown import set_pool
from typing import Optional,Callable,Any
from torchvision.models.resnet import ResNet,_resnet,conv1x1,conv3x3


# Circular padding (class)
class cpad(nn.Module):
  def __init__(self, pad):
    super(cpad,self).__init__()
    self.pad = pad
  def forward(self,x):
    return F.pad(x, pad = self.pad, mode = 'circular')
  def extra_repr(self):
    return ("pad={pad}".format(pad = self.pad))


# Replace and initialize conv
def replace_conv(in_ch,out_ch,kernel_size,
                 padding,padding_mode,init,
                 bias=False,stride=1):
  c=nn.Conv2d(in_channels=in_ch,
              out_channels=out_ch,
              kernel_size=kernel_size,
              padding=padding,
              padding_mode=padding_mode,
              bias=bias,
              stride=stride)
  if init:
    nn.init.kaiming_normal_(c.weight,
                            mode='fan_out',
                            nonlinearity='relu')
  return c


# Replace and initialize pool
def replace_pool(p,in_ch,out_ch,
                 kernel_size,padding,padding_mode,
                 init,bn,swap_conv_pool=False):
  # Conv
  c=nn.Conv2d(in_ch,
              out_ch,
              kernel_size=kernel_size,
              padding=padding,
              padding_mode=padding_mode,
              bias=False)
  if init:
    # Kaiming init.
    nn.init.kaiming_normal_(c.weight,
                            mode='fan_out',
                            nonlinearity='relu')

  if bn:
    # Include BN
    b=nn.BatchNorm2d(out_ch)
    if init:
      # Constant init.
      nn.init.constant_(b.weight,1)
      nn.init.constant_(b.bias,0)
    if swap_conv_pool: s=nn.Sequential(c,b,p) # Pool applied last
    else: s=nn.Sequential(p,c,b)
  else:
    if swap_conv_pool: s=nn.Sequential(c,p)
    else: s=nn.Sequential(p,c)
  return s


# Custom BasicBlock
class BasicBlockCustom(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, global_ret_prob=False) -> Tensor:
        # TODO: Generalize to any sequential length
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.stride>1 and self.ret_prob:
          # Return feats and prob
          if self.swap_conv_pool:
            # conv->pool
            out=self.conv2[0](out)
            out,_p=self.conv2[1](x=out,ret_prob=True)
          else:
            # pool->conv
            out,_p=self.conv2[0](x=out,ret_prob=True)
            out=self.conv2[1](out)
          out=self.bn2(out)
          if self.downsample is not None:
            # Pass original input and prob to downsample
            if self.forward_pool_method=="LPS" and self.training:
              # Train: If LPS, prob is first element of tuple
              p = _p[0] 
            else:
              p = _p
            identity=self.downsample[0](x=x,prob=p)
            identity=self.downsample[1](identity)
            identity=self.downsample[2](identity)
        else:
          # Original pipeline
          out = self.conv2(out)
          out = self.bn2(out)
          if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if global_ret_prob:
          # Train: Return feats and probability-logits tuple
          # Test: return feats and logits
          return out,_p
        return out


class BottleneckCustom(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BottleneckCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, global_ret_prob=False) -> Tensor:
        # TODO: Generalize to any sequential length
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.stride>1 and self.ret_prob:
          # Return feats and prob
          if self.swap_conv_pool:
            # conv->pool
            out=self.conv3[0](out)
            out,_p=self.conv3[1](x=out,ret_prob=True)
          else:
            # pool->conv
            out,_p=self.conv3[0](x=out,ret_prob=True)
            out=self.conv3[1](out)
          out=self.bn3(out)
          if self.downsample is not None:
            # Pass original input and prob to downsample
            if self.forward_pool_method=="LPS" and self.training:
              # Train: If LPS, prob is first element of tuple
              p = _p[0] 
            else:
              p = _p
            identity=self.downsample[0](x=x,prob=p)
            identity=self.downsample[1](identity)
            identity=self.downsample[2](identity)
        else:
          # Original pipeline
          out = self.conv3(out)
          out = self.bn3(out)
          if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if global_ret_prob:
          # Train: Return feats and probability-logits tuple
          # Test: return feats and logits
          return out,_p
        return out


# Core ResNet18, fixed shortcut via BasicBlockCustom
def resnet18_fs(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', BasicBlockCustom, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


# Core ResNet50, fixed shortcut via BottleneckCustom
def resnet50_fs(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', BottleneckCustom, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


# Core ResNet101, fixed shortcut via BottleneckCustom
def resnet101_fs(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet("resnet101", BottleneckCustom, [3, 4, 23, 3], pretrained, progress, **kwargs)


# Custom ResNet18 (CIFAR10, ImageNet)
class ResNet18Custom(AbstractBaseClassifierModel):
  def __init__(self,input_shape,num_classes,
               padding_mode='zeros',learning_rate=0.1,pooling_layer=nn.AvgPool2d,
               extras_model=None,**kwargs):
    super().__init__(**kwargs)
 
    # log hyperparameters
    self.save_hyperparameters()
    self.learning_rate=learning_rate

    # Model-specific extras
    self.swap_conv_pool=extras_model['swap_conv_pool']
    self.inc_conv1_support=extras_model['inc_conv1_support']
    self.apply_maxpool=extras_model['apply_maxpool']
    self.ret_prob=extras_model['ret_prob']
    self.logits_channels=extras_model['logits_channels'] if 'logits_channels' in extras_model.keys()\
      else False
    self.conv1_stride=extras_model['conv1_stride'] if 'conv1_stride' in extras_model.keys()\
      else False
    self.maxpool_zpad=extras_model['maxpool_zpad'] if 'maxpool_zpad' in extras_model.keys()\
      else False
    self.forward_pool_method = extras_model['forward_pool_method'] if 'forward_pool_method' in extras_model.keys()\
      else 'LPS'
    self.maxpool_no_antialias = extras_model['maxpool_no_antialias'] if 'maxpool_no_antialias' in extras_model.keys()\
      else True
    print("[ResNet18Custom] self.maxpool_no_antialias: {s}".format(s=self.maxpool_no_antialias))

    # ResNet18 model with fixed shortcut
    self.core=resnet18_fs()

    # Modify Conv2d padding attribute
    for layer in self.core.modules():
      if isinstance(layer,nn.Conv2d):
        layer.padding_mode=padding_mode

    # Pass extra to BasicBlock
    for i in range(len(self.core.layer1)):
      self.core.layer1[i].ret_prob= self.ret_prob
      self.core.layer1[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer1[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer2)):
      self.core.layer2[i].ret_prob= self.ret_prob
      self.core.layer2[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer2[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer3)):
      self.core.layer3[i].ret_prob= self.ret_prob
      self.core.layer3[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer3[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer4)):
      self.core.layer4[i].ret_prob= self.ret_prob
      self.core.layer4[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer4[i].forward_pool_method = self.forward_pool_method

    if pooling_layer is None:
      # Keep original pool
      if self.inc_conv1_support:
        # ImageNet/Imagenette: Conv1 unmodified
        pass
      else:
        # CIFAR10: Replace conv1, k=3
        self.core.conv1=replace_conv(in_ch=3,
                                     out_ch=64,
                                     kernel_size=3,
                                     padding=1,
                                     padding_mode=padding_mode,
                                     init=True)

      if self.apply_maxpool:
        # ImageNet/Imagenette: Maxpool unmodified
        pass
      else:
        # CIFAR10: Remove maxpool
        self.core.maxpool=nn.Sequential()
    else:
      # Replace pool
      # Logits model channels
      if self.logits_channels:
        maxpool_h_ch = self.logits_channels["maxpool"]
        layer2_h_ch = self.logits_channels["layer2"]
        layer3_h_ch = self.logits_channels["layer3"]
        layer4_h_ch = self.logits_channels["layer4"]
      else:
        maxpool_h_ch = 64
        layer2_h_ch = 128
        layer3_h_ch = 256
        layer4_h_ch = 512

      if self.inc_conv1_support:
        # ImageNet/Imagenette: Update conv1 stride
        conv1_stride=2 if self.conv1_stride else 1
        self.core.conv1=replace_conv(in_ch=3,
                                     out_ch=64,
                                     kernel_size=7,
                                     padding=3,
                                     padding_mode=padding_mode,
                                     stride=conv1_stride,
                                     init=True)

      else:
        # CIFAR10: Replace conv1, k=3
        self.core.conv1=replace_conv(in_ch=3,
                                     out_ch=64,
                                     kernel_size=3,
                                     padding=1,
                                     padding_mode=padding_mode,
                                     init=True)

      if self.apply_maxpool:
        # ImageNet/Imagenette: Replace maxpool stride by custom pool
        _maxpool = []
        if self.conv1_stride:
          # Conv1 stride applied already
          pass
        else:
          # Replace conv1 stride by custom pool
          _maxpool.append(set_pool(
            pooling_layer=pooling_layer,
            p_ch=64,
            h_ch=maxpool_h_ch,
            no_antialias=self.maxpool_no_antialias,
          ))
        if self.maxpool_zpad:
          _maxpool.append(nn.ZeroPad2d((0,1,0,1)))
        else:
          _maxpool.append(cpad(pad=[0,1,0,1]))
        _maxpool.append(nn.MaxPool2d(kernel_size=2,
                        stride=1))
        _maxpool.append(set_pool(pooling_layer=pooling_layer,
                        p_ch=64,
                        h_ch=maxpool_h_ch))
        self.core.maxpool=nn.Sequential(*_maxpool)
      else:
        # CIFAR10: Remove maxpool
        self.core.maxpool=nn.Sequential()

      # Replace stride [layer2, layer3, layer4]
      # Set main branch pool
      print("[ResNet18Custom] pooling_layer: {s}".format(s=pooling_layer))
      p2_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=128,
                    h_ch=layer2_h_ch)
      p3_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=256,
                    h_ch=layer3_h_ch)
      p4_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=512,
                    h_ch=layer4_h_ch)

      # Replace and init. layers
      self.core.layer2[0].conv1=replace_conv(in_ch=64,
                                             out_ch=128,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer2[0].conv2=replace_pool(p=p2_1,
                                             in_ch=128,
                                             out_ch=128,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)
      self.core.layer3[0].conv1=replace_conv(in_ch=128,
                                             out_ch=256,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer3[0].conv2=replace_pool(p=p3_1,
                                             in_ch=256,
                                             out_ch=256,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)
      self.core.layer4[0].conv1=replace_conv(in_ch=256,
                                             out_ch=512,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer4[0].conv2=replace_pool(p=p4_1,
                                             in_ch=512,
                                             out_ch=512,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)

      # Set shortcut branch pool
      # No component selection, indices precomputed.
      # Input channels still passed in case antialiasing is applied.
      # https://github.com/pytorch/vision/blob/863e904e4165fe42950c355325a93198d56e4271/torchvision/models/resnet.py#L78
      p2_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=64,
                    use_get_logits=False)
      p3_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=128,
                    use_get_logits=False)
      p4_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=256,
                    use_get_logits=False)

      # Replace and init. layers
      # Ksize=1, no padding required
      self.core.layer2[0].downsample=replace_pool(p=p2_2,
                                                  in_ch=64,
                                                  out_ch=128,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)
      self.core.layer3[0].downsample=replace_pool(p=p3_2,
                                                  in_ch=128,
                                                  out_ch=256,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)
      self.core.layer4[0].downsample=replace_pool(p=p4_2,
                                                  in_ch=256,
                                                  out_ch=512,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)

    # Replace head
    self.core.fc= nn.Linear(512,num_classes)

  def forward(self,x):
    out=self.core(x)
    out=F.log_softmax(out,dim=1)
    return out


# Custom ResNet50 (ImageNet)
class ResNet50Custom(AbstractBaseClassifierModel):
  def __init__(self,input_shape,num_classes,
               padding_mode='zeros',learning_rate=0.1,pooling_layer=nn.AvgPool2d,
               extras_model=None,**kwargs):
    super().__init__(**kwargs)
 
    # log hyperparameters
    self.save_hyperparameters()
    self.learning_rate=learning_rate

    # Model-specific extras
    self.logits_channels=extras_model['logits_channels']
    self.conv1_stride=extras_model['conv1_stride']
    self.maxpool_zpad=extras_model['maxpool_zpad']
    self.swap_conv_pool=extras_model['swap_conv_pool']
    self.inc_conv1_support=extras_model['inc_conv1_support']
    self.apply_maxpool=extras_model['apply_maxpool']
    self.ret_prob=extras_model['ret_prob']
    self.forward_pool_method = extras_model['forward_pool_method'] if 'forward_pool_method' in extras_model.keys()\
      else 'LPS'

    # ResNet50 model with fixed shortcut
    self.core=resnet50_fs()

    # Modify Conv2d padding attribute
    for layer in self.core.modules():
      if isinstance(layer,nn.Conv2d):
        layer.padding_mode=padding_mode

    # Pass extras to BasicBlock
    for i in range(len(self.core.layer1)):
      self.core.layer1[i].ret_prob= self.ret_prob
      self.core.layer1[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer1[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer2)):
      self.core.layer2[i].ret_prob= self.ret_prob
      self.core.layer2[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer2[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer3)):
      self.core.layer3[i].ret_prob= self.ret_prob
      self.core.layer3[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer3[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer4)):
      self.core.layer4[i].ret_prob= self.ret_prob
      self.core.layer4[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer4[i].forward_pool_method = self.forward_pool_method

    if pooling_layer is None:
      # Keep original pool
      pass
    else:
      # Replace pool
      # Logits model channels
      if self.logits_channels:
        maxpool_h_ch = self.logits_channels["maxpool"]
        layer2_h_ch = self.logits_channels["layer2"]
        layer3_h_ch = self.logits_channels["layer3"]
        layer4_h_ch = self.logits_channels["layer4"]
      else:
        maxpool_h_ch = 64
        layer2_h_ch = 128
        layer3_h_ch = 256
        layer4_h_ch = 512

      if self.inc_conv1_support:
        # ImageNet/Imagenette: Update conv1 stride
        conv1_stride=2 if self.conv1_stride else 1
        self.core.conv1=replace_conv(in_ch=3,
                                     out_ch=64,
                                     kernel_size=7,
                                     padding=3,
                                     padding_mode=padding_mode,
                                     stride=conv1_stride,
                                     init=True)

      if self.apply_maxpool:
        # ImageNet/Imagenette: Replace maxpool stride by custom pool
        _maxpool = []
        if self.conv1_stride:
          # Conv1 stride applied already
          pass
        else:
          # Replace conv1 stride by custom pool
          _maxpool.append(set_pool(pooling_layer=pooling_layer,
                                   p_ch=64,
                                   h_ch=maxpool_h_ch,
                                   no_antialias=True))
        if self.maxpool_zpad:
          _maxpool.append(nn.ZeroPad2d((0,1,0,1)))
        else:
          _maxpool.append(cpad(pad=[0,1,0,1]))
        _maxpool.append(nn.MaxPool2d(kernel_size=2,
                        stride=1))
        _maxpool.append(set_pool(pooling_layer=pooling_layer,
                        p_ch=64,
                        h_ch=maxpool_h_ch))
        self.core.maxpool=nn.Sequential(*_maxpool)

      # Replace stride [layer2, layer3, layer4]
      # Set main branch pool
      p2_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=128,
                    h_ch=layer2_h_ch)
      p3_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=256,
                    h_ch=layer3_h_ch)
      p4_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=512,
                    h_ch=layer4_h_ch)

      # Replace and init. layers
      self.core.layer2[0].conv2=replace_conv(in_ch=128,
                                             out_ch=128,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer2[0].conv3=replace_pool(p=p2_1,
                                             in_ch=128,
                                             out_ch=512,
                                             kernel_size=1,
                                             padding=0,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)
      self.core.layer3[0].conv2=replace_conv(in_ch=256,
                                             out_ch=256,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer3[0].conv3=replace_pool(p=p3_1,
                                             in_ch=256,
                                             out_ch=1024,
                                             kernel_size=1,
                                             padding=0,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)
      self.core.layer4[0].conv2=replace_conv(in_ch=512,
                                             out_ch=512,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer4[0].conv3=replace_pool(p=p4_1,
                                             in_ch=512,
                                             out_ch=2048,
                                             kernel_size=1,
                                             padding=0,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)

      # Set shortcut branch pool
      # No component selection, indices precomputed
      # https://github.com/pytorch/vision/blob/863e904e4165fe42950c355325a93198d56e4271/torchvision/models/resnet.py#L78
      p2_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=256,
                    use_get_logits=False)
      p3_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=512,
                    use_get_logits=False)
      p4_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=1024,
                    use_get_logits=False)

      # Replace and init. layers
      # Ksize=1, no padding required
      self.core.layer2[0].downsample=replace_pool(p=p2_2,
                                                  in_ch=256,
                                                  out_ch=512,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)
      self.core.layer3[0].downsample=replace_pool(p=p3_2,
                                                  in_ch=512,
                                                  out_ch=1024,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)
      self.core.layer4[0].downsample=replace_pool(p=p4_2,
                                                  in_ch=1024,
                                                  out_ch=2048,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)

    # Replace head
    self.core.fc= nn.Linear(2048,num_classes)

  def forward(self,x):
    out=self.core(x)
    out=F.log_softmax(out,dim=1)
    return out


# Custom ResNet101 (ImageNet)
class ResNet101Custom(AbstractBaseClassifierModel):
  def __init__(self,input_shape,num_classes,
               padding_mode='zeros',learning_rate=0.1,pooling_layer=nn.AvgPool2d,
               extras_model=None,**kwargs):
    super().__init__(**kwargs)
 
    # log hyperparameters
    self.save_hyperparameters()
    self.learning_rate=learning_rate

    # Model-specific extras
    self.logits_channels = extras_model['logits_channels']
    self.conv1_stride=extras_model['conv1_stride']
    self.maxpool_zpad = extras_model['maxpool_zpad']
    self.swap_conv_pool = extras_model['swap_conv_pool']
    self.inc_conv1_support = extras_model['inc_conv1_support']
    self.apply_maxpool = extras_model['apply_maxpool']
    self.ret_prob = extras_model['ret_prob']
    #self.forward_pool_method = extras_model['forward_pool_method']
    self.forward_pool_method = extras_model['forward_pool_method'] if 'forward_pool_method' in extras_model.keys()\
      else 'LPS'

    # ResNet101 model with fixed shortcut
    self.core=resnet101_fs()

    # Modify Conv2d padding attribute
    for layer in self.core.modules():
      if isinstance(layer,nn.Conv2d):
        layer.padding_mode=padding_mode

    # Pass extra to BottleNeck
    for i in range(len(self.core.layer1)):
      self.core.layer1[i].ret_prob= self.ret_prob
      self.core.layer1[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer1[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer2)):
      self.core.layer2[i].ret_prob= self.ret_prob
      self.core.layer2[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer2[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer3)):
      self.core.layer3[i].ret_prob= self.ret_prob
      self.core.layer3[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer3[i].forward_pool_method = self.forward_pool_method
    for i in range(len(self.core.layer4)):
      self.core.layer4[i].ret_prob= self.ret_prob
      self.core.layer4[i].swap_conv_pool= self.swap_conv_pool
      self.core.layer4[i].forward_pool_method = self.forward_pool_method

    if pooling_layer is None:
      # Keep original pool
      pass
    else:
      # Replace pool
      # Logits model channels
      if self.logits_channels:
        maxpool_h_ch = self.logits_channels["maxpool"]
        layer2_h_ch = self.logits_channels["layer2"]
        layer3_h_ch = self.logits_channels["layer3"]
        layer4_h_ch = self.logits_channels["layer4"]
      else:
        maxpool_h_ch = 64
        layer2_h_ch = 128
        layer3_h_ch = 256
        layer4_h_ch = 512

      if self.inc_conv1_support:
        # ImageNet/Imagenette: Replace conv1 stride
        conv1_stride=2 if self.conv1_stride else 1
        self.core.conv1=replace_conv(in_ch=3,
                                     out_ch=64,
                                     kernel_size=7,
                                     padding=3,
                                     padding_mode=padding_mode,
                                     stride=conv1_stride,
                                     init=True)

      if self.apply_maxpool:
        # ImageNet/Imagenette: Replace maxpool stride by custom pool
        _maxpool = []
        if self.conv1_stride:
          # Conv1 stride applied already
          pass
        else:
          # Replace conv1 stride by custom pool
          _maxpool.append(set_pool(pooling_layer=pooling_layer,
                                   p_ch=64,
                                   h_ch=maxpool_h_ch,
                                   no_antialias=True))
        if self.maxpool_zpad:
          _maxpool.append(nn.ZeroPad2d((0,1,0,1)))
        else:
          _maxpool.append(cpad(pad=[0,1,0,1]))
        _maxpool.append(nn.MaxPool2d(kernel_size=2,
                        stride=1))
        _maxpool.append(set_pool(pooling_layer=pooling_layer,
                        p_ch=64,
                        h_ch=maxpool_h_ch))
        self.core.maxpool=nn.Sequential(*_maxpool)

      # Replace stride [layer2, layer3, layer4]
      # Set main branch pool
      p2_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=128,
                    h_ch=layer2_h_ch)
      p3_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=256,
                    h_ch=layer3_h_ch)
      p4_1=set_pool(pooling_layer=pooling_layer,
                    p_ch=512,
                    h_ch=layer4_h_ch)

      # Replace and init. layers
      self.core.layer2[0].conv2=replace_conv(in_ch=128,
                                             out_ch=128,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer2[0].conv3=replace_pool(p=p2_1,
                                             in_ch=128,
                                             out_ch=512,
                                             kernel_size=1,
                                             padding=0,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)
      self.core.layer3[0].conv2=replace_conv(in_ch=256,
                                             out_ch=256,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer3[0].conv3=replace_pool(p=p3_1,
                                             in_ch=256,
                                             out_ch=1024,
                                             kernel_size=1,
                                             padding=0,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)
      self.core.layer4[0].conv2=replace_conv(in_ch=512,
                                             out_ch=512,
                                             kernel_size=3,
                                             padding=1,
                                             padding_mode=padding_mode,
                                             init=True)
      self.core.layer4[0].conv3=replace_pool(p=p4_1,
                                             in_ch=512,
                                             out_ch=2048,
                                             kernel_size=1,
                                             padding=0,
                                             padding_mode=padding_mode,
                                             swap_conv_pool= self.swap_conv_pool,
                                             init=True,
                                             bn=False)

      # Set shortcut branch pool
      # No component selection, indices precomputed
      # https://github.com/pytorch/vision/blob/863e904e4165fe42950c355325a93198d56e4271/torchvision/models/resnet.py#L78
      p2_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=256,
                    use_get_logits=False)
      p3_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=512,
                    use_get_logits=False)
      p4_2=set_pool(pooling_layer=pooling_layer,
                    p_ch=1024,
                    use_get_logits=False)

      # Replace and init. layers
      # Ksize=1, no padding required
      self.core.layer2[0].downsample=replace_pool(p=p2_2,
                                                  in_ch=256,
                                                  out_ch=512,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)
      self.core.layer3[0].downsample=replace_pool(p=p3_2,
                                                  in_ch=512,
                                                  out_ch=1024,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)
      self.core.layer4[0].downsample=replace_pool(p=p4_2,
                                                  in_ch=1024,
                                                  out_ch=2048,
                                                  kernel_size=1,
                                                  padding=0,
                                                  padding_mode=padding_mode,
                                                  init=True,
                                                  bn=True)

    # Replace head
    self.core.fc= nn.Linear(2048,num_classes)

  def forward(self,x):
    out=self.core(x)
    out=F.log_softmax(out,dim=1)
    return out

