#!/bin/bash

source /opt/conda/bin/activate
conda activate base

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py \
 -c configs/rtdetr/rtdetr_r50vd_6x_EFID2023_oadkpt.yml --fleet --eval
