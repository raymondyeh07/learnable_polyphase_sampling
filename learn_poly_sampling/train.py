import os
import json
import torch
from torch import nn
import clargs.debug_opts
import clargs.train_opts
import clargs.logger_opts
from functools import partial
import pytorch_lightning as pl
from data import get_datamodule
from absl import app, flags, logging
from utils.core_utils import loadcfg
from utils.logger_utils import set_logger,set_tags
from pytorch_lightning.callbacks import LearningRateMonitor
from callbacks import(
  PrintBufferCallback,AutoResumeState,OneEpochStop,
)
from models import(
  get_model,get_available_models,get_available_optimizers,
  get_optimizer,get_available_schedulers,get_scheduler,
  get_available_param_schedulers,get_param_scheduler,
)
from layers import(
  PolyphaseInvariantDown2D,max_p_norm,LPS,
  get_pool_method,get_available_pool_methods,get_available_logits_model,
  get_available_antialias,
)


AVAILABLE_OPTIMIZERS = get_available_optimizers()
AVAILABLE_SCHEDULERS = get_available_schedulers()
AVAILABLE_PARAM_SCHEDULERS = get_available_param_schedulers()
AVAILABLE_POOL_METHODS = get_available_pool_methods()
AVAILABLE_MODELS = get_available_models()
AVAILABLE_LOGITS_MODEL = get_available_logits_model()
AVAILABLE_ANTIALIAS = get_available_antialias()
FLAGS = flags.FLAGS

# Misc
flags.DEFINE_integer('seed',42,'Set random seed for consistent experiments')
flags.DEFINE_boolean('dryrun',False,'Run sanity check only on 1 batch of each split')
flags.DEFINE_boolean('autoresume',False,'enables autoresume from last checkpoint if previous run was incomplete')
flags.DEFINE_string('autoresume_statefile','.train_incomplete',"state file used by autoresume feature")
flags.DEFINE_boolean('oneepoch',False,'Stops training after one epoch regardless of max epochs')
flags.DEFINE_integer('precision',32,'precision to use for training')
flags.DEFINE_boolean('deterministic',False,'Lightning trainer reproducibility flag.')
flags.DEFINE_string('resume_cp',None,'Checkpoint path to resume training.')

# Dataset
flags.DEFINE_enum('aug_method','circular',['basic','shift','circular','randomresized'],"Which type of augmentation to apply")
flags.DEFINE_integer('batchsize',32,'batchsize')
flags.DEFINE_string('dataset','cifar10','dataset to train on: {cifar10 (default), imagenet}')
flags.DEFINE_string('dataroot','../datasets','where to find the data')
flags.DEFINE_float('val_split',None,'Percentage of training samples used for validation')

# Model
flags.DEFINE_enum('model', 'basic', AVAILABLE_MODELS, 'Model name: {basic (default), ResNet18Custom}')
flags.DEFINE_enum('pool_method', 'max_2_norm', AVAILABLE_POOL_METHODS, 'Pooling method: {max_2_norm (default), avgpool, LPS, Decimation, skip}')
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

# Train hyperparams
flags.DEFINE_integer('epochs',50,'number of epochs')
flags.DEFINE_integer('warmup_epochs',0,'number of warmup epochs')
flags.DEFINE_integer('accumulate_grad_batches',1,'Batch accumulation')
flags.DEFINE_enum('optimizer','SGD',AVAILABLE_OPTIMIZERS,'which optimizer to use')
flags.DEFINE_string('optimizer_cfg',None,'JSON file for optimizer config dictionary')
flags.DEFINE_float('learning_rate',0.1,'Learning rate')
flags.DEFINE_enum('lr_scheduler',None,AVAILABLE_SCHEDULERS,'Learning rate schedulers')
flags.DEFINE_string('lr_scheduler_cfg',None,'JSON file for lr scheduler config dictionary')
flags.DEFINE_enum('temperature_scheduler',None,AVAILABLE_PARAM_SCHEDULERS,'gumbel temperature scheduler')
flags.DEFINE_string('temperature_scheduler_cfg',None,'JSON file for gumbel temperature scheduler config dictionary')

# Anti aliasing
flags.DEFINE_enum('antialias_mode','skip',AVAILABLE_ANTIALIAS,'Antialiasing method: {skip (default) LowPassFilter, DDAC}')
flags.DEFINE_integer('antialias_size',None,'Antialiasing kernel size: {None (default) 2, 3, 5}')
flags.DEFINE_string('antialias_padding','same','Antialiasing padding size: {same (default), valid}.')
flags.DEFINE_string('antialias_padding_mode','circular','Antialiasing padding mode: {Circular (default), reflect, replicate, constant}.')
flags.DEFINE_integer('antialias_group',8,'DDAC channel groups: {1, 2, 3, 4, 8 (default), 16}')


def main(argv):
  if len(argv)>1:
    print('Unprocessed args:',argv[1:])

  # Check antialias args
  if FLAGS.antialias_mode!="skip":
    assert(FLAGS.antialias_size is not None)
  if FLAGS.antialias_size is not None:
    assert(FLAGS.antialias_mode!="skip")

  # Check no antialias selection
  if FLAGS.selection_noantialias:
    assert FLAGS.pool_method in ["LPS","max_2_norm"],\
    "Phase selection before antialiasing valid only for LPS and max_2_norm pool methods."
    assert FLAGS.antialias_mode!="skip",\
    "Phase selection before antialiasing valid only if an antialiasing filter is set."

  # Trainer root dir
  if FLAGS.logger=="tb":
    logdir=os.path.join(FLAGS.logdir,"TbLogger")
  elif FLAGS.logger=="neptune":
    logdir=os.path.join(FLAGS.logdir,"NeptuneLogger")
  else:
    raise ValueError("Undefined logger. Check 'logger' input argument.")

  # Set lr monitor
  lr_monitor = LearningRateMonitor(
    logging_interval='epoch',
    log_momentum=True,
  )

  dm = get_datamodule(FLAGS.dataset)(
    batch_size=FLAGS.batchsize,
    data_dir=FLAGS.dataroot,
    aug_method=FLAGS.aug_method,
    val_split=FLAGS.val_split,
  )
  dm.prepare_data()
  dm.setup()

  pool_layer = get_pool_method(FLAGS.pool_method, FLAGS)
  logits_channels = loadcfg(FLAGS.logits_channels) if FLAGS.logits_channels else None
  optimizer_kwargs = loadcfg(FLAGS.optimizer_cfg) if FLAGS.optimizer_cfg else {}

  if FLAGS.lr_scheduler:
    scheduler = {
      'scheduler':get_scheduler(FLAGS.lr_scheduler),
      'scheduler_kwargs': loadcfg(FLAGS.lr_scheduler_cfg),
    }
  else:
    scheduler = {}

  if FLAGS.temperature_scheduler:
    param_scheduler = {
      'param_scheduler': get_param_scheduler(FLAGS.temperature_scheduler),
      'param_scheduler_kwargs': loadcfg(FLAGS.temperature_scheduler_cfg)
    }
  else:
    param_scheduler = {}

  # Model-specific flags
  if FLAGS.model=='ResNet18Custom':
    extras_model = {
      'logits_channels': logits_channels,
      'maxpool_zpad': FLAGS.maxpool_zpad,
      'swap_conv_pool': FLAGS.swap_conv_pool,
      'conv1_stride': FLAGS.conv1_stride,
      'inc_conv1_support': True if FLAGS.dataset in ['imagenet','imagenette'] else False,
      'apply_maxpool': True  if FLAGS.dataset in ['imagenet','imagenette'] else False,
      'ret_prob': True if FLAGS.pool_method in ['max_2_norm','LPS'] else False,
      'ret_logits': True if FLAGS.pool_method=='LPS' else False,
      'forward_pool_method': FLAGS.pool_method,
    }
  elif FLAGS.model in ['ResNet50Custom','ResNet101Custom']:
    assert FLAGS.dataset in ['imagenet','imagenette'],\
    "ResNet50 and ResNet101 models only available for ImageNet."
    extras_model = {
      'logits_channels': logits_channels,
      'maxpool_zpad': FLAGS.maxpool_zpad,
      'swap_conv_pool': FLAGS.swap_conv_pool,
      'conv1_stride': FLAGS.conv1_stride,
      'inc_conv1_support': True,
      'apply_maxpool': True,
      'ret_prob': True if FLAGS.pool_method in ['max_2_norm','LPS'] else False,
      'ret_logits': True if FLAGS.pool_method=='LPS' else False,
      'forward_pool_method': FLAGS.pool_method,
    }
  else:
      extras_model = None

  model = get_model(FLAGS.model)(
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

  # Set logger
  name=f'{FLAGS.model}-{FLAGS.optimizer}-{FLAGS.pool_method}'
  params=FLAGS.flag_values_dict()
  tags=set_tags(FLAGS)
  mc = pl.callbacks.ModelCheckpoint(
    filename='{epoch}-{val_acc:.3f}',
    monitor='val_acc',
    save_top_k=5,
    mode='max',
  )

  ar_cb = AutoResumeState(
    mc,
    enabled=FLAGS.autoresume,
    resume_cp=FLAGS.resume_cp,
    max_epochs = FLAGS.epochs,
    single_epoch_mode = FLAGS.oneepoch,
    state_file=FLAGS.autoresume_statefile,
  )

  # Set random seed
  if ar_cb.checkpoint_dir is None:
    print("Not autoresuming. Random seed set to: ",FLAGS.seed)
    pl.utilities.seed.seed_everything(FLAGS.seed)
  else:
    # Autoresuming, update seed based on epoch number
    _epoch = ar_cb.get_epoch()
    _seed = FLAGS.seed + _epoch + 1
    print("Autoresuming epoch (counting from 0): {s}".format(s=_epoch))
    print("Random seed set to : {s}".format(s=seed))
    pl.utilities.seed.seed_everything(_seed)

  logger= set_logger(
    logger=FLAGS.logger,
    logdir=logdir,
    name=name,
    params=params,
    tags=tags,
    version=ar_cb.resume_version,
    upload_source_files=FLAGS.upload_source_files,
    upload_stdout=FLAGS.upload_stdout,
    upload_stderr=FLAGS.upload_stderr,
    send_hardware_metrics=FLAGS.send_hardware_metrics,
  )

  cb=[mc,lr_monitor,ar_cb,OneEpochStop()] if FLAGS.oneepoch else\
    [mc,lr_monitor,ar_cb]

  if FLAGS.debug:
    logging.info('-------------------DEBUG PRINTS ENABLED--------------------')
    cb.append(PrintBufferCallback('gumbel_tau'))

  trainer = pl.Trainer(
    max_epochs=FLAGS.epochs,
    gpus=FLAGS.gpus,
    accelerator=FLAGS.accelerator,
    accumulate_grad_batches=FLAGS.accumulate_grad_batches,
    precision=FLAGS.precision,
    logger=logger,
    default_root_dir=logdir,
    callbacks=cb,
    fast_dev_run=FLAGS.dryrun,
    resume_from_checkpoint=ar_cb.resume_checkpoint_file,
    limit_train_batches=16 if FLAGS.tinysubset else 1.0,
    limit_val_batches=8 if FLAGS.tinysubset else 1.0,
    limit_test_batches=4 if FLAGS.tinysubset else 1.0,
    num_sanity_val_steps=0 if FLAGS.oneepoch else 2,
    deterministic=FLAGS.deterministic,
  )

  if FLAGS.debug:
    logging.info('-------------------TRAINER CREATED-------------------------')
    print(trainer.callbacks)

  trainer.fit(model,dm)
  if not FLAGS.dryrun and trainer.current_epoch == FLAGS.epochs-1:
    trainer.test(datamodule=dm)


if __name__ == '__main__':
  app.run(main)
