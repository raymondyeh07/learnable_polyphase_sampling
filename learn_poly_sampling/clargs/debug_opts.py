from absl import flags

# Debugging
flags.DEFINE_bool('debug', False, 'Enable spammy debugging printouts')
flags.DEFINE_bool('tinysubset', False, 'Train on a tiny subset (16 batches in train, 8 batches in val) for debugging/testing purposes')


