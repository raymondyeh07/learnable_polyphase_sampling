#!/bin/sh
root="./.."
logdir="./../../logs"
cpdir="./../checkpoints"
python3 ${root}/eval.py\
  --logdir ${logdir}\
  --dataroot ${root}/../datasets/ILSVRC2012\
  --dataset imagenet\
  --model ResNet101Custom\
  --batchsize 64\
  --logger neptune\
  --eval_mode class_accuracy\
  --shift_seed 21\
  --checkpoint ${cpdir}/classification/ResNet101_LPS_LPF3_zero_randomresized.ckpt
