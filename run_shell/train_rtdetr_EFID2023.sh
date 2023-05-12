#!/bin/bash

# source /opt/conda/bin/activate
# conda activate base

export CUDA_VISIBLE_DEVICES=0,1
python3 -m paddle.distributed.launch --gpus 0,1 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_oad_kpts.yml --fleet --eval