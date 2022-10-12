import json
from absl import app, flags, logging
from data import get_segmentation_datamodule

from callbacks import PrintBufferCallback, AutoResumeState, OneEpochStop
from functools import partial
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import os
from utils.logger_utils import set_logger, set_tags
from utils.segmentation_utils import update_extras_model
from models.base_segmentation import DDAC_MODEL_MAP, DDACSegmentation
from models import (
    get_available_optimizers, get_optimizer,
    get_available_schedulers, get_scheduler,
    get_available_param_schedulers, get_param_scheduler,
)
from models import get_available_models as get_available_backbones
from models import get_model as get_backbone

from layers import (
    get_pool_method, get_available_pool_methods,
    get_available_logits_model, get_available_antialias, get_unpool_method,
)

import clargs.logger_opts  # noqa: F401
import clargs.debug_opts   # noqa: F401
import clargs.train_opts   # noqa: F401


AVAILABLE_OPTIMIZERS = get_available_optimizers()
AVAILABLE_SCHEDULERS = get_available_schedulers()
AVAILABLE_PARAM_SCHEDULERS = get_available_param_schedulers()
AVAILABLE_POOL_METHODS = get_available_pool_methods()
AVAILABLE_MODELS = list(DDAC_MODEL_MAP.keys()) # DDAC model names
AVAILABLE_BACKBONES = get_available_backbones() # LPS classifier models
AVAILABLE_LOGITS_MODEL = get_available_logits_model()
AVAILABLE_ANTIALIAS = get_available_antialias()

FLAGS = flags.FLAGS

# Misc
flags.DEFINE_integer('seed', 42, 'Set random seed for consistent experiments')
flags.DEFINE_boolean('dryrun', False, 'Run sanity check only on 1 batch of each split')
flags.DEFINE_boolean('autoresume', False, 'enables autoresume from last checkpoint if previous run was incomplete')
flags.DEFINE_string('autoresume_statefile', '.train_incomplete', "state file used by autoresume feature")
flags.DEFINE_boolean('oneepoch', False, 'Stops training after one epoch regardless of max epochs')
flags.DEFINE_integer('precision', 32, 'precision to use for training')
flags.DEFINE_string('resume_cp', None, 'Checkpoint path to resume training.')

# Dataset
flags.DEFINE_string('dataset', 'voc2012_aug', 'Dataset to train on')
flags.DEFINE_string('dataroot', '../datasets/', 'Root directory of dataset')
flags.DEFINE_integer('batchsize', 16, 'batchsize')

# Model
flags.DEFINE_enum('model', AVAILABLE_MODELS[0], AVAILABLE_MODELS, 'Model to train')
flags.DEFINE_enum('loss_type', 'focal', ['crossentropy', 'focal'], 'loss function to use')

# These model flags are only used when model is resnet_custom
flags.DEFINE_enum('backbone', AVAILABLE_BACKBONES[0], AVAILABLE_BACKBONES,
                  'Backbone model to use. This flag is only when model is resnet_custom')
flags.DEFINE_enum('pool_method', 'max_2_norm', AVAILABLE_POOL_METHODS, 'pooling method: {max_2_norm (default), avgpool, LPS, decimation, skip}')
flags.DEFINE_boolean('circular_pad', False, 'Use circular padding instead of default zeros padding where applicable')
flags.DEFINE_enum('logits_model', 'LPSLogitLayers', AVAILABLE_LOGITS_MODEL, 'Model to compute logits')
flags.DEFINE_string('logits_channels', None, 'JSON file with logits model number of channels')
flags.DEFINE_string('LPS_pad', 'circular', 'Logits layer padding scheme {zeros, reflect, replicate, circular}')
flags.DEFINE_boolean('LPS_gumbel', False, 'Keep pooling in train mode (gumbel-smax sampling) during validation.')
flags.DEFINE_boolean('LPS_convex', False, '[Test, val] output a convex phase combination based on logits at test time.')
flags.DEFINE_boolean('LPS_train_convex', False, '[Train] output a convex phase combination based on logits at test time.')
flags.DEFINE_boolean('selection_noantialias', False, 'Compute polyphase selection before applying lowpass filtering. Valid only if antialias enabled.')
flags.DEFINE_integer('pool_k', 2, 'Pooling kernel size for {avgpool, Decimation}.')
flags.DEFINE_boolean('swap_conv_pool', False, '{ResNet18Custom, ResNet50Custom, ResNet101Custom}: Use conv->pool instead of pool->conv.')
flags.DEFINE_boolean('conv1_stride', False, '{ResNet18Custom, ResNet50Custom, ResNet101Custom}: Keep stride on first conv layer.')
flags.DEFINE_boolean('maxpool_zpad', False, '{ResNet18Custom, ResNet50Custom, ResNet101Custom}: Use zero instead of circular padding on first maxpool.')

flags.DEFINE_string('backbone_weights', None, 'Path to pretrained weights for backbone model')

# Overrides
flags.DEFINE_boolean('update_extras_model', False, 'Add items to original extras_model dictionary. Valid only is backbone_weights are passed.')
flags.DEFINE_boolean('override_extras_model', False, 'Override extras_model dictionary with new values. Valid only if update_extras_model is True.')
flags.DEFINE_boolean('override_pool_layer', False, 'Override pool layer with new configuration. Might fail when pool_method results in incompatible weights.')

# Train hyperparams
flags.DEFINE_integer('epochs', 50, 'number of epochs')
flags.DEFINE_integer('warmup_epochs', 0, 'number of warmup epochs')
flags.DEFINE_integer('accumulate_grad_batches', 1, 'Batch accumulation')
flags.DEFINE_enum('optimizer', 'SGD', AVAILABLE_OPTIMIZERS, 'which optimizer to use')
flags.DEFINE_string('optimizer_cfg', None, 'JSON file for optimizer config dictionary')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
flags.DEFINE_float('backbone_lr_scale', 0.1, 'backbone learning rate multiplier')
flags.DEFINE_float('pool_par_lr_scale', 0.1, 'pooling parameters learning rate multiplier')
flags.DEFINE_enum('lr_scheduler', None, AVAILABLE_SCHEDULERS, 'Learning rate schedulers')
flags.DEFINE_string('lr_scheduler_cfg', None, 'JSON file for lr scheduler config dictionary')
flags.DEFINE_enum('temperature_scheduler', None, AVAILABLE_PARAM_SCHEDULERS, 'gumbel temperature scheduler')
flags.DEFINE_string('temperature_scheduler_cfg', None, 'JSON file for gumbel temperature scheduler config dictionary')
flags.DEFINE_enum('scheduler_interval', 'epoch', ['epoch', 'step'], 'Scheduler step interval')

# Anti aliasing
flags.DEFINE_enum('antialias_mode', 'skip', AVAILABLE_ANTIALIAS, 'Antialiasing method: {skip (default) LowPassFilter, DDAC}')
flags.DEFINE_integer('antialias_size', None, 'Antialiasing kernel size: {None (default) 2, 3, 5}')
flags.DEFINE_string('antialias_padding', 'same', 'Antialiasing padding size: {same (default), valid}.')
flags.DEFINE_string('antialias_padding_mode', 'circular', 'Antialiasing padding mode: {Circular (default), reflect, replicate, constant}.')
flags.DEFINE_integer('antialias_group', 8, 'DDAC channel groups: {1, 2, 3, 4, 8 (default), 16}')
flags.DEFINE_boolean('antialias_separable_conv', False, 'Use separable convolutions instead of regular convolutions')

# Unpool
flags.DEFINE_boolean('unpool', False, 'Use unpooling layers on classifier. Unpooling method matches pool_method input argument.')
flags.DEFINE_boolean('unpool_get_samples', False, 'Compute unpool phase probability by applying gumbel-softmax over pre-computed pool logits.')
flags.DEFINE_boolean('LPS_u_convex', False, '[Test, val] output a convex upsampling combination based on logits at test time.')
flags.DEFINE_enum('unpool_antialias_mode', 'skip', AVAILABLE_ANTIALIAS, 'Antialiasing method: {skip (default) LowPassFilter, DDAC}')
flags.DEFINE_integer('unpool_antialias_size', None, 'Antialiasing kernel size: {None (default) 2, 3, 5}')
flags.DEFINE_integer('unpool_antialias_scale', 2, 'Antialiasing kernel scale (ideally must be equivalent to stride factor)')
flags.DEFINE_string('unpool_antialias_padding', 'same', 'Antialiasing padding size: {same (default), valid}.')
flags.DEFINE_string('unpool_antialias_padding_mode', 'circular', 'Antialiasing padding mode: {Circular (default), reflect, replicate, constant}.')
flags.DEFINE_integer('unpool_antialias_group', 8, 'DDAC channel groups: {1, 2, 3, 4, 8 (default), 16}')
flags.DEFINE_boolean('unpool_antialias_separable_conv', False, 'Use separable convolutions instead of regular convolutions')
flags.DEFINE_string('classifier_padding_mode', 'circular', 'Classifer padding mode: {Circular (default), reflect, replicate, constant}.')


# Segmentation Specific
flags.DEFINE_integer('output_stride', 16, 'Output stride in Deeplabv3(plus) heads')


def main(argv):
    # Check extras_model args
    if FLAGS.update_extras_model:
      assert FLAGS.backbone_weights is not None,\
      "Update extras model selected, but no backbone checkpoint loaded."

    # Check unpool args
    if 'unpool' in FLAGS.model:
        assert FLAGS.unpool,\
        "Segmentation model with unpooling selected but no unpooling method assigned."
    if FLAGS.unpool:
        assert 'unpool' in FLAGS.model,\
        "Unpooling method assigned but no segmentation model with unpooling selected."

    # Check antialias args
    if FLAGS.antialias_mode!="skip":
        assert(FLAGS.antialias_size is not None)
    if FLAGS.antialias_size is not None:
        assert(FLAGS.antialias_mode!="skip")

    # Trainer root dir
    if FLAGS.logger == "tb":
        logdir = os.path.join(FLAGS.logdir, "TbLogger")
    elif FLAGS.logger == "neptune":
        logdir = os.path.join(FLAGS.logdir, "NeptuneLogger")
    else:
        raise ValueError("Undefined logger. Check 'logger' input argument.")

    # Set lr monitor
    lr_monitor = LearningRateMonitor(logging_interval=FLAGS.scheduler_interval,
                                     log_momentum=True)

    # Dataset
    if 'voc' in FLAGS.dataset and FLAGS.unpool and\
        FLAGS.pool_method in ['LPS','max_2_norm']:
      # Crop_size= 512
      print("crop_size = 512")
      dm = get_segmentation_datamodule(FLAGS.dataset)(
          batch_size=FLAGS.batchsize,
          data_dir=FLAGS.dataroot,
          val_batch_size=1,
          crop_val=True,
          crop_size=512,
      )
    else:
      # Default crop_size
      dm = get_segmentation_datamodule(FLAGS.dataset)(
          batch_size=FLAGS.batchsize,
          data_dir=FLAGS.dataroot,
      )

    dm.prepare_data()
    dm.setup()

    # Pool and unpool layers
    pool_layer = get_pool_method(FLAGS.pool_method, FLAGS)
    unpool_layer = get_unpool_method(unpool=FLAGS.unpool,
                                     pool_method=FLAGS.pool_method,
                                     antialias_mode=FLAGS.unpool_antialias_mode,
                                     antialias_size=FLAGS.unpool_antialias_size,
                                     antialias_padding=FLAGS.unpool_antialias_padding,
                                     antialias_padding_mode=FLAGS.unpool_antialias_padding_mode,
                                     antialias_group=FLAGS.unpool_antialias_group,
                                     antialias_scale=FLAGS.unpool_antialias_scale,
                                     get_samples=FLAGS.unpool_get_samples,
                                     LPS_u_convex=FLAGS.LPS_u_convex)

    def _loadcfg(fn):
        try:
            with open(fn, 'r') as f:
                return json.load(f)
        except TypeError as e:
            logging.error(f'Filename to config file required. {e}')
            raise

    if FLAGS.logits_channels:
        logits_channels = _loadcfg(FLAGS.logits_channels)
    else:
        logits_channels = None
    if FLAGS.optimizer_cfg:
        optimizer_kwargs = _loadcfg(FLAGS.optimizer_cfg)
    else:
        optimizer_kwargs = {}

    if FLAGS.lr_scheduler:
        scheduler = {
            'scheduler':get_scheduler(FLAGS.lr_scheduler),
            'scheduler_kwargs': _loadcfg(FLAGS.lr_scheduler_cfg)
        }
    else:
        scheduler = {}

    if FLAGS.temperature_scheduler:
        param_scheduler = {
            'param_scheduler': get_param_scheduler(FLAGS.temperature_scheduler),
            'param_scheduler_kwargs': _loadcfg(FLAGS.temperature_scheduler_cfg)
        }
    else:
        param_scheduler = {}

    # Model-specific args
    if FLAGS.backbone == 'ResNet18Custom':
        extras_model = {
            'logits_channels': logits_channels,
            'maxpool_zpad': FLAGS.maxpool_zpad,
            'swap_conv_pool': FLAGS.swap_conv_pool,
            'conv1_stride': FLAGS.conv1_stride,
            'inc_conv1_support': True,
            'apply_maxpool': True,
            'ret_prob': True if FLAGS.pool_method in ['max_2_norm', 'LPS'] else False,
            'ret_logits': True if FLAGS.pool_method=='LPS' else False,
            'forward_pool_method': FLAGS.pool_method,
        }
    elif FLAGS.backbone in ['ResNet50Custom', 'ResNet101Custom']:
        extras_model = {
            'logits_channels': logits_channels,
            'maxpool_zpad': FLAGS.maxpool_zpad,
            'swap_conv_pool': FLAGS.swap_conv_pool,
            'conv1_stride': FLAGS.conv1_stride,
            'inc_conv1_support': True,
            'apply_maxpool': True,
            'ret_prob': True if FLAGS.pool_method in ['max_2_norm', 'LPS'] else False,
            'ret_logits': True if FLAGS.pool_method=='LPS' else False,
            'forward_pool_method': FLAGS.pool_method,
        }
    else:
        extras_model = None

    # build resnet_lps backbone
    if 'resnet_lps' in FLAGS.model:
        if FLAGS.backbone_weights:
            logging.info(f'Loading weights from {FLAGS.backbone_weights}')
            if FLAGS.update_extras_model:
              # Add items from default to checkpoint extras_model dict
              # Compatibility: Existing items in the checkpoint dictionary cannot be modified.
              # Only new items from the default extras_model dictionary can be added to it.
              _extras_model = update_extras_model(backbone_weights=FLAGS.backbone_weights,
                                                  extras_model=extras_model,
                                                  override=FLAGS.override_extras_model,
                                                 )
              if FLAGS.override_pool_layer:
                  backbone = get_backbone(FLAGS.backbone).load_from_checkpoint(
                      FLAGS.backbone_weights,
                      extras_model=_extras_model,
                      pooling_layer=pool_layer,
                  )
              else:
                  backbone = get_backbone(FLAGS.backbone).load_from_checkpoint(
                      FLAGS.backbone_weights,
                      extras_model=_extras_model,
                  )
            else:
              # Keep checkpoint extras_model dict
              backbone = get_backbone(FLAGS.backbone).load_from_checkpoint(FLAGS.backbone_weights)
        else:
            backbone = get_backbone(FLAGS.backbone)(
                dm.size(),
                dm.num_classes,
                padding_mode='circular' if FLAGS.circular_pad else 'zeros',
                learning_rate=FLAGS.learning_rate,
                optimizer=get_optimizer(FLAGS.optimizer),
                optimizer_kwargs=optimizer_kwargs,
                pooling_layer=pool_layer,
                extras_model=extras_model,
                warmup_epochs=FLAGS.warmup_epochs,
                **scheduler,
                **param_scheduler
            )
    else:
        backbone = None

    model = DDACSegmentation(
        loss_type=FLAGS.loss_type,
        model_name=FLAGS.model,
        num_classes=dm.num_classes,
        learning_rate=FLAGS.learning_rate,
        backbone_lr_scale=FLAGS.backbone_lr_scale,
        pool_par_lr_scale=FLAGS.pool_par_lr_scale,
        separable_conv=FLAGS.antialias_separable_conv,
        output_stride=FLAGS.output_stride,
        filter_size=FLAGS.antialias_size,
        groups=FLAGS.antialias_group,
        ignore_index=dm.ignore_index,
        backbone=backbone,
        unpool_layer=unpool_layer,
        optimizer=get_optimizer(FLAGS.optimizer),
        optimizer_kwargs=optimizer_kwargs,
        warmup_epochs=FLAGS.warmup_epochs,
        scheduler_interval=FLAGS.scheduler_interval,
        classifier_padding_mode=FLAGS.classifier_padding_mode,
        **scheduler,
        **param_scheduler)

    # Set logger
    name = f'{FLAGS.dataset}_{FLAGS.model}_{FLAGS.pool_method}'
    params = FLAGS.flag_values_dict()

    mc = pl.callbacks.ModelCheckpoint(
        filename='{epoch}-{val_seg_Mean_IoU:.3f}',
        monitor='val_seg_Mean_IoU',
        save_top_k=5,
        mode='max',
    )

    ar_cb = AutoResumeState(mc,
                            enabled=FLAGS.autoresume,
                            resume_cp=FLAGS.resume_cp,
                            max_epochs = FLAGS.epochs,
                            single_epoch_mode = FLAGS.oneepoch,
                            state_file=FLAGS.autoresume_statefile)

    # Set random seed
    if ar_cb.checkpoint_dir is None:
      # Not autoresuming, set seed as usual
      print("Not autoresuming. Random seed set to: ",FLAGS.seed)
      pl.utilities.seed.seed_everything(FLAGS.seed)
    else:
      # Autoresuming, refresh seed based on epoch number
      # Epoch count starts at 0
      _epoch = ar_cb.get_epoch()
      _seed = FLAGS.seed + _epoch + 1
      print("Autoresuming epoch (counting from 0): ",_epoch)
      print("Random seed set to : ",_seed)
      pl.utilities.seed.seed_everything(_seed)

    print("ar_cb.resume_version: ",ar_cb.resume_version)
    logger = set_logger(logger=FLAGS.logger,
                        name=name,
                        logdir=logdir,
                        tags=set_tags(FLAGS),
                        params=params,
                        project_name="cig-uiuc/learnable-polyphase-segmentation",
                        version=ar_cb.resume_version,
                        upload_source_files=FLAGS.upload_source_files,
                        upload_stdout=FLAGS.upload_stdout,
                        upload_stderr=FLAGS.upload_stderr,
                        send_hardware_metrics=FLAGS.send_hardware_metrics)

    cb = [mc, lr_monitor, ar_cb]

    if FLAGS.oneepoch:
        cb.append(OneEpochStop())

    trainer = pl.Trainer(
        gpus=FLAGS.gpus,
        max_epochs=FLAGS.epochs,
        logger=logger,
        accelerator=FLAGS.accelerator,
        precision=FLAGS.precision,
        accumulate_grad_batches=FLAGS.accumulate_grad_batches,
        default_root_dir=logdir,
        callbacks=cb,
        fast_dev_run=FLAGS.dryrun,
        resume_from_checkpoint=ar_cb.resume_checkpoint_file,
        limit_train_batches=16 if FLAGS.tinysubset else 1.0,
        limit_val_batches=8 if FLAGS.tinysubset else 1.0,
        limit_test_batches=4 if FLAGS.tinysubset else 1.0,
        num_sanity_val_steps=0 if FLAGS.oneepoch else 2,
    )

    trainer.fit(model, dm)

    trainer.test()


if __name__ == '__main__':
    app.run(main)
