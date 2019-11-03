# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/home/yuria_utsumi/deepcluster/tiny-imagenet-200/"
#DATA="/datasets01/imagenet_full_size/061417/"
MODELROOT="/home/yuria_utsumi/deepcluster/results/checkpoints/"
#MODELROOT="${HOME}/deepcluster_models"
MODEL="${MODELROOT}/checkpoint_0.0.pth.tar"
#MODEL="${MODELROOT}/alexnet/checkpoint.pth.tar"
EXP="${HOME}/deepcluster_exp/linear_classif"

PYTHON="/home/yuria_utsumi/miniconda3/bin/python"
#PYTHON="${HOME}/test/conda/bin/python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 3 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12
