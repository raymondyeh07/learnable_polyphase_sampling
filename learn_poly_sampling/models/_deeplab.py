import torch
from torch import nn
from torch.nn import functional as F
from layers.polyup import set_unpool,unpool_multistage
from .thirdparty.network.utils import _SimpleSegmentationModel
from .thirdparty.network._deeplab import DeepLabHeadV3Plus

class DeepLabV3Unpool(_SimpleSegmentationModel):
    def __init__(self,backbone,classifier,
                 num_classes,unpool_layer=None):
        super().__init__(backbone=backbone,
                         classifier=classifier)
        self.unpool_layer = unpool_layer

        if self.unpool_layer is not None:
          # Set unpooling
          # ([b,num_classes,56,56] -> [b,num_classes,224,224]),
          # scaling=4 (2 unpooling + antialias filters)
          self.unpool = nn.ModuleList()
          for i in range(2):
            self.unpool.append(set_unpool(unpool_layer=self.unpool_layer,
                                          p_ch=num_classes))

    def forward(self, x):
        input_shape = x.shape[-2:]
        if self.unpool_layer is not None:
          features,p_dict = self.backbone(x)
          x = self.classifier(features,
                              p_dict=p_dict)
          sf = input_shape[0] // x.shape[-2]
          x = unpool_multistage(x=x,
                                x_layer='low_level',
                                p_dict=p_dict,
                                scale_factor=sf,
                                unpool_layer=self.unpool)
        else:
          features = self.backbone(x)
          x = self.classifier(features)
          x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


class DeepLabHeadV3PlusUnpool(DeepLabHeadV3Plus):
    def __init__(self, in_channels, low_level_channels,
                 num_classes, aspp_dilate=[12, 24, 36], unpool_layer=None,
                 padding_mode='zeros'):
        super().__init__(in_channels=in_channels,
                         low_level_channels=low_level_channels,
                         num_classes=num_classes,
                         aspp_dilate=aspp_dilate)
        self.unpool_layer = unpool_layer
        self.padding_mode = padding_mode

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False, padding_mode=self.padding_mode),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels,
                         aspp_dilate,
                         padding_mode=self.padding_mode)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False, padding_mode=self.padding_mode),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, padding_mode=self.padding_mode)
        )
        self._init_weight()

        if self.unpool_layer is not None:
          # Set unpooling. ([b,256,7,7] -> [b,256,56,56]),
          # scaling=8 (3 unpooling + antialias filters)
          self.unpool = nn.ModuleList()
          for i in range(3):
            self.unpool.append(set_unpool(unpool_layer=self.unpool_layer,
                                          p_ch=256))

    def forward(self, feature, p_dict=None):
        low_level_feature = self.project( feature['low_level'])
        output_feature = self.aspp(feature['out'])
        if self.unpool_layer is not None:
          sf = low_level_feature.shape[2] // output_feature.shape[2]
          output_feature = unpool_multistage(x=output_feature,
                                             x_layer='out',
                                             p_dict=p_dict,
                                             scale_factor=sf,
                                             unpool_layer=self.unpool)
        else:
          output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 dilation, padding_mode='zeros'):
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 padding_mode='zeros'):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,
                      out_channels,
                      1,
                      bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates,
                 padding_mode='zeros'):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, padding_mode=padding_mode))
        modules.append(ASPPConv(in_channels, out_channels, rate2, padding_mode=padding_mode))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels, padding_mode=padding_mode))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False,
                      padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

