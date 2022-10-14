from .cifar10 import CIFAR10DataModule
from .imagenet import ImageNetDataModule
from .imagenette import ImagenetteDataModule
from .voc import VOCSegmentationDataModule
from .cityscapes import CityscapesDataModule


available_modules = {
    'cifar10': CIFAR10DataModule,
    'imagenet': ImageNetDataModule,
    'imagenette': ImagenetteDataModule,
}


available_segmentation_datasets = {
    'voc2012_aug': VOCSegmentationDataModule,
    'cityscapes': CityscapesDataModule,
}


def get_datamodule(name):
    return available_modules[name]


def get_segmentation_datamodule(name):
    return available_segmentation_datasets[name]
