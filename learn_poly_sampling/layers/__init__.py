from .polydown import(PolyphaseInvariantDown2D,fixed_component,max_p_norm,
                      LPS,Decimation)
from .polyup import PolyphaseInvariantUp2D,max_p_norm_u,LPS_u
from .lowpass_filter import LowPassFilter,DDAC
from .lps_logit_layers import(LPSLogitLayersV2,LPSLogitLayersSkip,SAInner,
                              SAInner_bn,GraphLogitLayers,ComponentPerceptron)
from functools import partial
from torch import nn

available_logits_models={
    'LPSLogitLayers':LPSLogitLayersV2,
    'LPSLogitLayersSkip':LPSLogitLayersSkip,
    'SAInner':SAInner,
    'SAInner_bn':SAInner_bn,
    'GraphLogitLayers':GraphLogitLayers,
    'ComponentPerceptron': ComponentPerceptron
    }

_available_antialias = ('LowPassFilter', 'DDAC', 'skip')

#TODO: this is not consistent with other API patterns
_available_pool_methods = ('max_2_norm', 'LPS', 'avgpool', 'Decimation', 'skip')
_available_unpool_methods={
    'max_2_norm': 'max_2_norm_u',
    'LPS': 'LPS_u',
    }

def get_available_pool_methods(): return _available_pool_methods

def get_pool_method(name, FLAGS):
    #different pool methods uses different flags, this needs cleaning up
    assert name in _available_pool_methods
    antialias_layer = get_antialias(antialias_mode=FLAGS.antialias_mode,
                                    antialias_size=FLAGS.antialias_size,
                                    antialias_padding=FLAGS.antialias_padding,
                                    antialias_padding_mode=FLAGS.antialias_padding_mode,
                                    antialias_group=FLAGS.antialias_group)
    pool_method = {
        'max_2_norm': partial(
            PolyphaseInvariantDown2D,
            component_selection=max_p_norm,
            antialias_layer=antialias_layer,
            selection_noantialias=FLAGS.selection_noantialias,
        ),
        'LPS': partial(
            PolyphaseInvariantDown2D,
            component_selection= LPS,
            get_logits=get_logits_model(FLAGS.logits_model),
            logits_pad=FLAGS.LPS_pad,
            comp_fix_train=FLAGS.LPS_gumbel,
            comp_train_convex=FLAGS.LPS_train_convex,
            comp_convex=FLAGS.LPS_convex,
            antialias_layer=antialias_layer,
            selection_noantialias=FLAGS.selection_noantialias,
        ),
        'Decimation': partial(
            Decimation,
            stride=FLAGS.pool_k,
            antialias_layer=antialias_layer,
        ),
        'avgpool': partial(
            nn.AvgPool2d,
            kernel_size=FLAGS.pool_k
        ),
        'skip': None
    }
    return pool_method[name]


def get_unpool_method(unpool,pool_method,antialias_mode,
                      antialias_size,antialias_padding,antialias_padding_mode,
                      antialias_group,antialias_scale,get_samples,LPS_u_convex=False):
    # Different pool methods uses different flags, this needs cleaning up
    name = _available_unpool_methods[pool_method] if unpool else 'skip'
    antialias_layer = get_antialias(antialias_mode=antialias_mode,
                                    antialias_size=antialias_size,
                                    antialias_padding=antialias_padding,
                                    antialias_padding_mode=antialias_padding_mode,
                                    antialias_group=antialias_group,
                                    antialias_scale=antialias_scale)
    unpool_method = {
        'max_2_norm_u': partial(
            PolyphaseInvariantUp2D,
            component_selection=max_p_norm_u,
            antialias_layer=antialias_layer,
        ),
        'LPS_u': partial(
            PolyphaseInvariantUp2D,
            component_selection=LPS_u,
            get_samples=get_samples,
            comp_convex=LPS_u_convex,
            antialias_layer=antialias_layer,
        ),
        'skip': None
    }
    return unpool_method[name]

def get_antialias(antialias_mode,antialias_size,antialias_padding,
                  antialias_padding_mode,antialias_group,antialias_scale=1):
    assert antialias_mode in _available_antialias
    antialias = {
        'LowPassFilter': partial(
            LowPassFilter,
            filter_size=antialias_size,
            filter_scale=antialias_scale,
            padding=antialias_padding,
            padding_mode=antialias_padding_mode,
        ),
        'DDAC': partial(
            DDAC,
            kernel_size=antialias_size,
            kernel_scale=antialias_scale,
            pad_type=antialias_padding_mode,
            group=antialias_group,
        ),
        'skip': None
    }
    return antialias[antialias_mode]

def get_logits_model(name):
    return available_logits_models[name]

def get_available_logits_model():
    return list(available_logits_models.keys())

def get_available_antialias(): return _available_antialias
