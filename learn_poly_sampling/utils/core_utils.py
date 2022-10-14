import json
def loadcfg(fn):
  try:
    with open(fn, 'r') as f: return json.load(f)
  except TypeError as e:
    logging.error(f'Filename to config file required. {e}')
    raise
