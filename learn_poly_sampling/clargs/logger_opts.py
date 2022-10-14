from absl import flags

# Logger config
flags.DEFINE_enum('logger','tb',['tb', 'neptune'], 
                  'Logger to use. Neptune logger requires NEPTUNE_API_TOKEN set.')
flags.DEFINE_boolean('upload_source_files',False,'Neptune upload source code.')
flags.DEFINE_boolean('upload_stdout',False,'Neptune upload shell standard output.')
flags.DEFINE_boolean('upload_stderr',False,'Neptune upload shell standard error.')
flags.DEFINE_boolean('send_hardware_metrics',False,'Neptune send hardware metrics.')

