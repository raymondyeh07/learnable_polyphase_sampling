from absl import flags

# Train config
flags.DEFINE_integer('gpus', 1, 'number of gpus to use')
flags.DEFINE_enum('accelerator', None, ['dp', 'ddp'], 'Distributed mode: {None (default), dp, ddp}')
flags.DEFINE_string('logdir', '../logs', 'logdir')

