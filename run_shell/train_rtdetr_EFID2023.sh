#!/bin/bash

source /opt/conda/bin/activate
conda activate base

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5 tools/train.py \
 -c configs/rtdetr/rtdetr_r50vd_6x_EFID2023_oad.yml --fleet --eval
