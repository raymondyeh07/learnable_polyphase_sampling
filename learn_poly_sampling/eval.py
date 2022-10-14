import os
import json
import torch
from torch import nn
from pathlib import Path
import clargs.train_opts
import clargs.logger_opts
from absl import app, flags
from models import get_model
from functools import partial
import pytorch_lightning as pl
from data import get_datamodule
from layers import(
  PolyphaseInvariantDown2D,
  max_p_norm,
  get_available_logits_model,
  get_available_antialias,
)
from utils.logger_utils import(
  set_logger,
  set_tags,read_tags,
)


AVAILABLE_LOGITS_MODEL = get_available_logits_model()
AVAILABLE_ANTIALIAS = get_available_antialias()
AVAILABLE_EVAL_MODE = ['class_accuracy','shift_consistency']
FLAGS = flags.FLAGS

# Misc
flags.DEFINE_boolean('dryrun',False,'Run sanity check only on 1 batch of each split')
flags.DEFINE_string('checkpoint',None,'Checkpoint to load')
flags.DEFINE_enum('eval_mode','class_accuracy',AVAILABLE_EVAL_MODE,'Test: {class_accuracy, shift_consistency}')
flags.mark_flag_as_required('checkpoint')

# Shift consistency
flags.DEFINE_integer('shift_seed',7,'Shift consistency: Random seed to generate shifts')
flags.DEFINE_integer('shift_max',32,'Shift consistency: Maximum offset shifts')
flags.DEFINE_integer('shift_samples',2,'Shift consistency: Number of samples to compare')
flags.DEFINE_integer('shift_patch_size',224,'Shift consistency: Image size to compute consistency')

# Dataset
flags.DEFINE_integer('batchsize',32,'Batchsize')
flags.DEFINE_string('dataset','cifar10','Eval dataset')
flags.DEFINE_string('dataroot','./../datasets','Data directory')

# Model
flags.DEFINE_string('model',None,'Model name: {basic (default), ResNet18Custom}')


def main(argv):
  if len(argv)>1:
    print('Unprocessed args:',argv[1:])

  # Trainer root dir
  if FLAGS.logger=="tb":
    logdir = os.path.join(FLAGS.logdir,"TbLogger")
  elif FLAGS.logger=="neptune":
    logdir = os.path.join(FLAGS.logdir,"NeptuneLogger")
  else:
    raise ValueError("Undefined logger. Check 'logger' input argument.") 

  # Dataset
  dm = get_datamodule(FLAGS.dataset)(
    batch_size=FLAGS.batchsize,
    data_dir=FLAGS.dataroot,
    base_center_crop=256 if FLAGS.eval_mode=='shift_consistency' and FLAGS.dataset in ['imagenet','imagenette']\
      else 224,
    aug_method='shift_consistency' if FLAGS.eval_mode=='shift_consistency' and FLAGS.dataset in ['imagenet','imagenette']\
      else 'basic'
  )
  dm.prepare_data()
  dm.setup()

  if FLAGS.dryrun:
    # Dryrun
    model = get_model(FLAGS.model)(
      dm.size(),
      dm.num_classes,
      pooling_layer=partial(nn.AvgPool2d,kernel_size=2),
    )
  else:
    # Load from checkpoint
    model = get_model(FLAGS.model).load_from_checkpoint(
      FLAGS.checkpoint,
      eval_mode=FLAGS.eval_mode,
      shift_seed=FLAGS.shift_seed,
      shift_max=FLAGS.shift_max,
      shift_samples=FLAGS.shift_samples,
      shift_patch_size=FLAGS.shift_patch_size,
    )

  # Set logger
  params=FLAGS.flag_values_dict()
  logger= set_logger(
    logger=FLAGS.logger,
    logdir=logdir,
    params=params,
    upload_source_files=FLAGS.upload_source_files,
    upload_stdout=FLAGS.upload_stdout,
    upload_stderr=FLAGS.upload_stderr,
    send_hardware_metrics=FLAGS.send_hardware_metrics,
  )
  trainer = pl.Trainer(
    gpus=FLAGS.gpus,
    logger=logger,
    default_root_dir=logdir,
    fast_dev_run=FLAGS.dryrun,
  )
  trainer.test(
    model,
    datamodule=dm,
  )


if __name__=='__main__':
  app.run(main)
