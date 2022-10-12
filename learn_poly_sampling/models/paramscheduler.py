import torch
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import logging


class _ParameterScheduler(Callback):
  def __init__(self, filter_name, log_name='tau', mode='on_batch', log_single_param=True, verbose=False):
    super().__init__()
    self.filter_name = filter_name
    self.state = {'counter': 0, 't': 0, 'offset': 0, 'milestone_idx': 0}
    self.log_name = log_name
    self.log_single = log_single_param
    if mode not in ['on_batch', 'on_epoch']:
      raise ValueError('Unsupported mode', mode)
    self.mode = mode
    self.verbose = verbose

  def on_load_checkpoint(self, trainer, pl_module, callback_state):
    self.state.update(callback_state)
    logging.info(f'Restored scheduler state: counter = {self.counter}, t = {self.t}, '
                 f'offset = {self.offset}, milestone_idx = {self.milestone_idx}')

  def on_save_checkpoint(self, trainer, pl_module, checkpoint):
    logging.info(f'Saving scheduler state: counter = {self.counter}, t = {self.t}, '
                 f'offset = {self.offset}, milestone_idx = {self.milestone_idx}')
    return self.state.copy()

  def on_train_batch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', outputs, batch, batch_idx, dataloader_idx) -> None:
    if self.mode == 'on_batch':
      self.step(trainer, pl_module)

  def on_train_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    if self.mode == 'on_epoch':
      self.step(trainer, pl_module)

  def get_next_param_val(self):
    raise NotImplementedError

  @property
  def counter(self):
    return self.state['counter']

  @property
  def t(self):
    return self.state['t']

  @property
  def offset(self):
    return self.state['offset']

  @property
  def milestone_idx(self):
    return self.state['milestone_idx']

  def param_list(self, pl_module):
    try:
      return self._param_list
    except AttributeError:
      self._param_list = [p for n, p in pl_module.named_buffers() if self.filter_name in n]
      return self._param_list

  def step(self, trainer, pl_module):
    self.state['counter'] += 1
    param_list = self.param_list(pl_module)
    if len(param_list) == 0:
      logging.warning('No parameters passed in. This scheduler will have no effect!')
      return
    self._prev_param_value = param_list[0]
    next_param_val = self.get_next_param_val()

    for p in param_list:
      p.set_(next_param_val)

    if self.log_single:
      trainer.logger.log_metrics({f'{self.log_name}': param_list[0]}, step=trainer.global_step)
    else:
      trainer.logger.log_metrics({f'{self.log_name}_{i}': v for i, v in enumerate(param_list)},
                                 step=trainer.global_step)

class StepDecay(_ParameterScheduler):
  def __init__(self, filter_name, step_size, gamma=0.1, min_value=0, verbose=False, **kwargs):
    self.step_size = step_size
    self.gamma = gamma
    self.min_val = min_value

    super().__init__(filter_name, verbose=verbose, **kwargs)

  def get_next_param_val(self):
    if not isinstance(self.min_val, torch.Tensor):
      self.min_val = torch.tensor(self.min_val).to(self._prev_param_value.device)
    if self.counter % self.step_size==0:
      next_val = max(self._prev_param_value * self.gamma, self.min_val)
      return next_val
    else:
      return self._prev_param_value

class MultiStep(_ParameterScheduler):
  def __init__(self, filter_name, milestones,
               temperatures, rate="zero", verbose=False,
               **kwargs):
    super().__init__(filter_name, verbose=verbose, **kwargs)
    self.milestones = milestones
    self.temperatures = temperatures

    # Decay rate
    self.rate = rate
    assert self.rate in ["zero","linear"]
    self.state['milestone_idx'] = 0

  def get_next_param_val(self):
    if self.rate == "zero":
      # zero-order hold
      try:
        if self.counter >= self.milestones[self.milestone_idx]:
          self.state['milestone_idx'] += 1
          next_val = self._prev_param_value/self._prev_param_value * self.temperatures[self.milestone_idx-1]
          return next_val
        else:
          return self._prev_param_value
      except IndexError:
        return self._prev_param_value
    elif self.rate == "linear":
      # linear interp
      try:
        if self.counter == 1:
          # Init temp and offset
          self.state['t'] = self._prev_param_value.clone()
          self.state['offset'] = 1
        if self.counter >= self.milestones[self.milestone_idx]:
          # Update milestone, temp and offset
          self.state['milestone_idx']+= 1
          self.state['t'] = self.temperatures[self.milestone_idx-1]
          self.state['offset'] = self.milestones[self.milestone_idx-1]

        # Decay
        self.s = (self.counter-self.offset)/(self.milestones[self.milestone_idx]-self.offset)
        next_val = (self._prev_param_value/self._prev_param_value-self.s)*self.t+self.s*self.temperatures[self.milestone_idx]
        return next_val
      except IndexError:
        # End of milestones, keep last value
        next_val = self._prev_param_value/self._prev_param_value*self.temperatures[self.milestone_idx-1]
        return next_val

