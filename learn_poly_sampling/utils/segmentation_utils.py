import torch
import logging


logger = logging.getLogger(__name__)

def update_extras_model(backbone_weights, extras_model, override=False):
  # Update cp
  cp = torch.load(backbone_weights)["hyper_parameters"]['extras_model']
  _extras_model = dict(list(extras_model.items()) + list(cp.items()))

  # Compatibility: Check if original items are unaltered
  logger.info("cp: {}".format(cp))
  logger.info("extras_model: {}".format(extras_model))
  logger.info("_extras_model: {}".format(_extras_model))
  if not override:
    assert _extras_model == extras_model,\
          "[Update_extras_model] Existing items in 'backbone_weights' extras_model do not match those in the "\
          "default extras_model dictionary. Make sure only new entries are added from the default dictionary "\
          "to the backbone checkpoint dictionary."
  else:
    for k,v in cp.items():
      if extras_model[k] != v:
        logger.warning("Overriding existing extras_model item {}: {} -> {}".format(k, v, extras_model[k]))
        _extras_model[k] = v
  logger.info("overridden _extras_model: {}".format(_extras_model))
  return _extras_model
