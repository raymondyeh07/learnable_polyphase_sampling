from .basic import BasicClassifier
from .resnet_custom import ResNet18Custom,ResNet50Custom,ResNet101Custom
from torch import optim
from torch.optim import lr_scheduler
from . import paramscheduler
from .thirdparty.utils.scheduler import PolyLR

from inspect import isclass

_available_classifiers = {
    'basic': BasicClassifier,
    'ResNet18Custom': ResNet18Custom,
    'ResNet50Custom': ResNet50Custom,
    'ResNet101Custom': ResNet101Custom
    }

_available_optimizers = {
    k:v for k, v in optim.__dict__.items() if isclass(v) and v is not optim.Optimizer and issubclass(v, optim.Optimizer)
}

_available_schedulers = {
        k:v for k, v in lr_scheduler.__dict__.items() if isclass(v) and v is not lr_scheduler._LRScheduler and issubclass(v, lr_scheduler._LRScheduler)
    }

_available_schedulers['PolyLR'] = PolyLR

_available_param_schedulars = {
        k:v for k, v in paramscheduler.__dict__.items() if isclass(v) and v is not paramscheduler._ParameterScheduler and issubclass(v, paramscheduler._ParameterScheduler)
}

def get_available_models():
    return list(_available_classifiers.keys())

def get_available_param_schedulers():
    return list(_available_param_schedulars.keys())

def get_param_scheduler(name):
    return _available_param_schedulars[name]

def get_available_schedulers():
    return list(_available_schedulers.keys())

def get_scheduler(name):
    return _available_schedulers[name]

def get_available_optimizers():
    return list(_available_optimizers.keys())

def get_optimizer(name):
    return _available_optimizers[name]

def get_model(name):
    return _available_classifiers[name]
