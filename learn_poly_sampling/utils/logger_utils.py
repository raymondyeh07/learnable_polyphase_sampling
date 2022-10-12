import os
import torch
from datetime import datetime
import pytorch_lightning.loggers as pl_loggers


# Set logger
def set_logger(
  logger,params,logdir,
  upload_source_files,upload_stdout,upload_stderr,
  send_hardware_metrics,project_name="project_name",
  version=None,name=None,tags=None,
):
  if logger=="tb":
    # Set Tensorboard
    ret=pl_loggers.TensorBoardLogger(save_dir=logdir,
                                     name=name,
                                     version=version,
                                    )
  elif logger=="neptune":
    # Set Neptune
    ret=set_neptune(name=name,
                    params=params,
                    tags=tags,
                    project_name=project_name,
                    version=version,
                    upload_source_files=upload_source_files,
                    upload_stdout=upload_stdout,
                    upload_stderr=upload_stderr,
                    send_hardware_metrics=send_hardware_metrics)
  else:
    raise ValueError("Undefined logger. Check 'logger' input argument.")
  return ret


# Set Neptune
def set_neptune(name,params,tags,version,
                project_name,
                upload_source_files=False,upload_stdout=False,upload_stderr=False,
                send_hardware_metrics=False):
  kwargs={"upload_stdout":upload_stdout,
          "upload_stderr":upload_stderr,
          "send_hardware_metrics":send_hardware_metrics}
  if not(upload_source_files):
    kwargs["upload_source_files"]=[]
  nl=pl_loggers.NeptuneLogger(api_key=os.environ["NEPTUNE_API_TOKEN"],
                              project_name=project_name,
                              experiment_name=name,
                              params=params,
                              tags=tags,
                              experiment_id=version,
                              **kwargs)
  return nl


def read_tags(FLAGS):
  # From filename
  # Cp fname structure: <logdir>/<logger>/<tags>/<ID>/checkpoints/<base>.ckpt
  _tags=FLAGS.checkpoint.rsplit('/',5)[-4]
  _tags=_tags.rsplit('-',3)
  if FLAGS.model is None:FLAGS.model=_tags[0]
  if FLAGS.optimizer is None:FLAGS.optimizer=_tags[1]
  if FLAGS.pool_method is None:FLAGS.pool_method=_tags[2]

  # From dictionary
  cp = torch.load(FLAGS.checkpoint)["hyper_parameters"]
  if FLAGS.pool_method == "LPS" and FLAGS.logits_model is None:
    # Get logits_model
    pooling_layer = cp["pooling_layer"]
    FLAGS.logits_model = pooling_layer.keywords["component_selection"].__name__
  if FLAGS.pool_method in ["max_2_norm","LPS","downsampling"] and\
     FLAGS.antialias_mode is None:
    # Get antialias_mode
    pooling_layer = cp["pooling_layer"]
    antialias = pooling_layer.keywords["antialias_layer"]
    if antialias is not None:
      FLAGS.antialias_mode = antialias.func.__name__
    else:
      FLAGS.antialias_mode = "skip"
  return


def set_tags(FLAGS,mode="train"):
  assert mode in ["train","test"]

  tags=[FLAGS.dataset, FLAGS.model, FLAGS.pool_method]
  if hasattr(FLAGS, "optimizer"):
    tags.append(FLAGS.model)

  if FLAGS.pool_method=="LPS":
    tags.append(FLAGS.logits_model)
  if FLAGS.antialias_mode!="skip":
    tags.append(FLAGS.antialias_mode)
  if mode=="test":
    tags.append("test")
  tags=[str(k) for k in tags]
  return tags
