#!/bin/sh
root="./.."
logdir="./../../logs"
cpdir="./../checkpoints"
python3 ${root}/eval.py\
  --logdir ${logdir}\
  --dataroot ${root}/../datasets/ILSVRC2012\
  --dataset imagenet\
  --model ResNet50Custom\
  --batchsize 64\
  --eval_mode class_accuracy\
  --shift_seed 7\
  --checkpoint ${cpdir}/classification/ResNet50_LPS_LPF3_zero_randomresized.ckpt
