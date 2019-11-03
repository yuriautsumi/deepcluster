# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/home/yuria_utsumi/deepcluster/training_data/"
#DIR="/home/yuria_utsumi/deepcluster/training_data/data/SpongeBobSquarePants-s1e01b-ReefBlower[Fester1500]"
#DIR="/home/yuria_utsumi/deepcluster/tiny-imagenet-200/train"
#DIR="/Users/yuriautsumi/6.867/deepcluster/tiny-imagenet-200/train"
#DIR="/datasets01/imagenet_full_size/061417/train"
PATH_FILE="/home/yuria_utsumi/deepcluster/training_data/train.txt"
ARCH="alexnet"
LR=0.05
WD=-5
K=200
#K=10000
WORKERS=12
EXP="./spongebob_results"
#EXP="/private/home/${USER}/test/exp"
PYTHON="/usr/lib/python3"
#PYTHON="/anaconda3/bin/python"
#PYTHON="/private/home/${USER}/test/conda/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 python3 spongebob_main.py ${DIR} --path_file ${PATH_FILE} --exp ${EXP} --arch ${ARCH} \
 --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
#CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
#  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
