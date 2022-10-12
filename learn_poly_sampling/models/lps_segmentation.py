import logging
from functools import partial

from .thirdparty.network.utils import IntermediateLayerGetter
from .thirdparty.network._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from ._deeplab_utils import IntermediateLayerGetterUnpool
from ._deeplab import DeepLabHeadV3PlusUnpool,DeepLabV3Unpool
                                         
from models import get_model as get_backbone
from layers import PolyphaseInvariantDown2D, LPS, get_logits_model


logger = logging.getLogger(__name__)


def get_default_lps_resnet(input_shape, num_classes):
    extras_model = {
        'logits_channels': None,
        'conv1_stride': False,
        'maxpool_zpad': False,
        'swap_conv_pool': False,
        'inc_conv1_support': True,
        'apply_maxpool': True,
        'ret_prob': True,
        'forward_pool_method': None,
    }
    get_logits = get_logits_model('LPSLogitLayers')
    pooling_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits)
    backbone_model = get_backbone('ResNet50Custom')(
        input_shape=(3, None, None),
        num_classes=num_classes,
        pooling_layer=pooling_layer,
        extras_model=extras_model,
    )
    return backbone_model


def lps_resnet_segmentation(
    name,
    backbone_model,
    num_classes,
    output_stride,
):
    """
    Create segmentation model using provided LPS ResNet backbone.
    Args:
        name (str): Name of the model.
        backbone_model (PL module): ResNet LPS backbone model. (PL modul
        num_classes (int): Number of classes to predict.
        output_stride (int): Output stride of the backbone model.
    """

    if backbone_model is None:
        logger.warning(
            "No backbone model provided. Using default ResNet LPS model."
        )
        backbone_model = get_default_lps_resnet(
            input_shape=(3, None, None),
            num_classes=num_classes,
        )
    backbone = backbone_model.core

    logger.info(f"Using backbone model: {type(backbone_model).__name__}")

    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilationi = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if "ResNet18" in type(backbone_model).__name__:
        inplanes = 512
        low_level_planes = 64
    else:
        inplanes = 2048
        low_level_planes = 256

    if 'deeplabv3plus' in name:
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif 'deeplabv3' in name:
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    else:
        raise ValueError('Unknown model name: {}'.format(name))
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    del backbone_model
    return model


def lps_resnet_segmentation_unpool(
    name,
    backbone_model,
    num_classes,
    output_stride,
    unpool_layer=None,
    classifier_padding_mode='zeros',
):
    if backbone_model is None:
        logger.warning(
            "No backbone model provided. Using default ResNet LPS model."
        )
        backbone_model = get_default_lps_resnet(
            input_shape=(3, None, None),
            num_classes=num_classes,
        )
    backbone = backbone_model.core

    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if "ResNet18" in type(backbone_model).__name__:
        inplanes = 512
        low_level_planes = 64
    else:
        inplanes = 2048
        low_level_planes = 256

    if 'deeplabv3plus' in name:
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3PlusUnpool(inplanes,
                                             low_level_planes,
                                             num_classes,
                                             aspp_dilate,
                                             unpool_layer=unpool_layer,
                                             padding_mode=classifier_padding_mode)
    elif 'deeplabv3' in name:
        raise ValueError("Unpooling currently not supported for 'deeplabv3'")
    else:
        raise ValueError('Unknown model name: {}'.format(name))
    backbone = IntermediateLayerGetterUnpool(backbone,
                                             return_layers=return_layers,
                                             unpool_layer=unpool_layer)
    model = DeepLabV3Unpool(backbone=backbone,
                            classifier=classifier,
                            unpool_layer=unpool_layer,
                            num_classes=num_classes)
    return model

