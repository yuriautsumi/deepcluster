# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

# First arg: DIR
# Second arg: LABELS (true if labels can be parsed from data directory, false otherwise)
# Third arg: K
# Fourth arg: EXPERIMENT (baseline, trivial_time)
# Fifth arg: WT_TIME (how much weight is placed on the times)

# Example:
# DIR="/home/jingweim/deepcluster/toy_dataset"
# LABELS=True
# K=3
# EXPERIMENT=baseline
# WT_TIME=1.0
# ./main_base.sh "/home/jingweim/deepcluster/toy_dataset" True 3 baseline 1.0

EXPERIMENT="${4}"
DIR="${1}"
LABELS=${2}
PATH_FILE="${1}/train.txt"
ARCH="alexnet"
LR=0.05
WD=-5
K=${3}
WT_TIME=${5}
WORKERS=12
EXP="./results"
PYTHON="/usr/lib/python3"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 python3 main_base.py ${DIR} --experiment ${EXPERIMENT} --labels ${LABELS} --path_file ${PATH_FILE} \
--weight_time ${WT_TIME} --exp ${EXP} --arch ${ARCH} --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
