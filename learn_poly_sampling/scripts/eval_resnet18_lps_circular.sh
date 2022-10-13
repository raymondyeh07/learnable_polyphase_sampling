#!/bin/sh
root="./.."
logdir="./../../logs"
cpdir="./../checkpoints"
python3 ${root}/eval.py\
  --logdir ${logdir}\
  --dataroot ${root}/../datasets/ILSVRC2012\
  --dataset imagenet\
  --model ResNet18Custom\
  --batchsize 64\
  --eval_mode class_accuracy\
  --shift_seed 42\
  --checkpoint ${cpdir}/classification/ResNet18_LPS_circular_basic.ckpt
